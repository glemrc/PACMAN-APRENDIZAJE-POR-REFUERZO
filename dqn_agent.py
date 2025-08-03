import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from dqn_network import DQNNetwork
from replay_buffer import ReplayBuffer, EpsilonScheduler, RewardShaper
from config import Config
import os

class DQNAgent:
    """
    Agente Deep Q-Network para Pac-Man
    Implementa DQN con mejoras opcionales como Double DQN y Prioritized Replay
    """
    
    def __init__(self, state_shape, n_actions, config=None, device=None):
        """
        Inicializa el agente DQN
        
        Args:
            state_shape: forma del estado de entrada (4, 84, 84)
            n_actions: número de acciones posibles
            config: configuración de hiperparámetros
            device: dispositivo para computación (cpu/cuda)
        """
        if config is None:
            config = Config()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.config = config
        self.device = device
        self.state_shape = state_shape
        self.n_actions = n_actions
        
        # Contadores
        self.step_count = 0
        self.episode_count = 0
        self.training_step = 0
        
        # Crear redes neuronales
        self.q_network = DQNNetwork(state_shape, n_actions).to(device)
        self.target_network = DQNNetwork(state_shape, n_actions).to(device)
        
        # Optimizador
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=config.LEARNING_RATE,
            eps=1e-4  # Epsilon para Adam (estabilidad numérica)
        )
        
        # Scheduler de learning rate opcional
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=100000, 
            gamma=0.5
        )
        
        # Buffer de experiencias
        self.memory = ReplayBuffer(
            capacity=config.BUFFER_SIZE,
            device=device,
            prioritized=config.USE_PRIORITIZED_REPLAY
        )
        
        # Scheduler de exploración
        self.epsilon_scheduler = EpsilonScheduler(
            start=config.EPSILON_START,
            end=config.EPSILON_END,
            decay_steps=config.EPSILON_DECAY_STEPS
        )
        
        # Utilidades opcionales
        if config.USE_REWARD_SHAPING:
            self.reward_shaper = RewardShaper()
        else:
            self.reward_shaper = None
        
        # Métricas de entrenamiento
        self.losses = []
        self.q_values = []
        self.episode_rewards = []
        
        # Inicializar target network
        self.update_target_network()
        
        print(f"✅ Agente DQN inicializado en dispositivo: {device}")
        print(f"🧠 Parámetros de la red: {self.count_parameters():,}")
    
    def count_parameters(self):
        """Cuenta el número total de parámetros del modelo"""
        return sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
    
    def act(self, state, training=True):
        """
        Selecciona una acción usando política epsilon-greedy
        
        Args:
            state: estado actual del juego
            training: si está en modo entrenamiento (usa epsilon)
            
        Returns:
            acción seleccionada
        """
        if training:
            epsilon = self.epsilon_scheduler.get_epsilon()
            if random.random() < epsilon:
                return random.randrange(self.n_actions)
        
        # Acción greedy usando la red neuronal
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            # Guardar Q-valores para monitoreo
            if training:
                self.q_values.append(q_values.max().item())
            
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done, info=None):
        """
        Almacena experiencia en el buffer de replay
        
        Args:
            state: estado actual
            action: acción tomada
            reward: recompensa recibida
            next_state: siguiente estado
            done: si el episodio terminó
            info: información adicional del entorno
        """
        # Aplicar reward shaping si está habilitado
        if self.reward_shaper and info:
            reward = self.reward_shaper.shape_reward(reward, info, done)
        
        # Almacenar en buffer
        self.memory.push(state, action, reward, next_state, done)
        
        # Actualizar contadores
        self.step_count += 1
        if training_ready := self.step_count >= self.config.LEARNING_STARTS:
            self.epsilon_scheduler.step()
    
    def learn(self):
        """
        Entrena la red neuronal con un batch de experiencias
        """
        if len(self.memory) < self.config.BATCH_SIZE:
            return None
        
        if self.step_count < self.config.LEARNING_STARTS:
            return None
        
        # Solo entrenar cada TRAIN_FREQ pasos
        if self.step_count % self.config.TRAIN_FREQ != 0:
            return None
        
        # Obtener batch de experiencias
        batch = self.memory.sample(self.config.BATCH_SIZE)
        loss = self._compute_loss(batch)
        
        # Optimización
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        self.lr_scheduler.step()
        
        # Actualizar target network periódicamente
        if self.training_step % self.config.TARGET_UPDATE_FREQ == 0:
            self.update_target_network()
        
        # Guardar métricas
        self.losses.append(loss.item())
        self.training_step += 1
        
        return loss.item()
    
    def _compute_loss(self, batch):
        """
        Calcula la pérdida para el entrenamiento
        
        Args:
            batch: batch de experiencias del buffer
            
        Returns:
            pérdida calculada
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Q-valores actuales
        current_q_values = self.q_network(states).gather(1, actions)
        
        if self.config.USE_DOUBLE_DQN:
            # Double DQN: usar red principal para selección, target para evaluación
            with torch.no_grad():
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
        else:
            # DQN clásico
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        
        # Calcular targets usando ecuación de Bellman
        target_q_values = rewards + (self.config.GAMMA * next_q_values * ~dones)
        
        # Pérdida MSE
        if 'weights' in batch:  # Prioritized replay
            weights = batch['weights']
            loss = (weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
            
            # Actualizar prioridades
            with torch.no_grad():
                td_errors = torch.abs(current_q_values - target_q_values).cpu().numpy().flatten()
                self.memory.update_priorities(batch['indices'], td_errors)
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        return loss
    
    def update_target_network(self):
        """Actualiza la red objetivo"""
        if self.config.SOFT_UPDATE:
            # Actualización suave
            tau = self.config.TAU
            for target_param, local_param in zip(self.target_network.parameters(), 
                                                 self.q_network.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        else:
            # Actualización dura (copia completa)
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath):
        """
        Guarda el modelo entrenado
        
        Args:
            filepath: ruta donde guardar el modelo
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon_scheduler.get_epsilon(),
            'losses': self.losses,
            'episode_rewards': self.episode_rewards
        }, filepath)
        
        print(f"💾 Modelo guardado en: {filepath}")
    
    def load_model(self, filepath):
        """
        Carga un modelo previamente entrenado
        
        Args:
            filepath: ruta del modelo a cargar
        """
        if not os.path.exists(filepath):
            print(f"❌ No se encontró el modelo en: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restaurar estado del entrenamiento
        self.step_count = checkpoint.get('step_count', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.losses = checkpoint.get('losses', [])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        
        # Restaurar epsilon
        if 'epsilon' in checkpoint:
            self.epsilon_scheduler.current_step = int(
                self.config.EPSILON_DECAY_STEPS * 
                (1 - (checkpoint['epsilon'] - self.config.EPSILON_END) / 
                 (self.config.EPSILON_START - self.config.EPSILON_END))
            )
        
        print(f"✅ Modelo cargado desde: {filepath}")
        print(f"📊 Episodio: {self.episode_count}, Pasos: {self.step_count}")
        return True
    
    def get_stats(self):
        """
        Retorna estadísticas del entrenamiento
        
        Returns:
            diccionario con métricas actuales
        """
        return {
            'episode': self.episode_count,
            'step': self.step_count,
            'epsilon': self.epsilon_scheduler.get_epsilon(),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_q_value': np.mean(self.q_values[-100:]) if self.q_values else 0,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'buffer_size': len(self.memory),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def reset_episode(self):
        """Reinicia variables para nuevo episodio"""
        self.episode_count += 1
        if self.reward_shaper:
            self.reward_shaper.reset()
    
    def set_eval_mode(self):
        """Pone el agente en modo evaluación (sin exploración)"""
        self.q_network.eval()
    
    def set_train_mode(self):
        """Pone el agente en modo entrenamiento"""
        self.q_network.train()
    
    def get_epsilon(self):
        """Retorna el valor actual de epsilon"""
        return self.epsilon_scheduler.get_epsilon()
    
    def print_network_info(self):
        """Imprime información detallada de la red"""
        print("\n🧠 INFORMACIÓN DE LA RED NEURONAL")
        print("="*40)
        print(f"Forma de entrada: {self.state_shape}")
        print(f"Número de acciones: {self.n_actions}")
        print(f"Parámetros totales: {self.count_parameters():,}")
        print(f"Dispositivo: {self.device}")
        
        # Información de capas
        print("\n📋 Arquitectura:")
        for name, module in self.q_network.named_modules():
            if len(list(module.children())) == 0:  # Solo capas finales
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    print(f"  {name}: {module} ({params:,} parámetros)")
        
        print("="*40)