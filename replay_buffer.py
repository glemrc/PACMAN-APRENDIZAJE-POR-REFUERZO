import numpy as np
import torch
import random
from collections import deque, namedtuple

# Definir la estructura de experiencia
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """
    Buffer de experiencias para Deep Q-Learning con Prioritized Experience Replay opcional
    """
    
    def __init__(self, capacity, device, prioritized=False, alpha=0.6):
        """
        Args:
            capacity: tamaño máximo del buffer
            device: dispositivo para tensores (cpu/cuda)
            prioritized: usar experiencia priorizada
            alpha: parámetro de priorización
        """
        self.capacity = capacity
        self.device = device
        self.prioritized = prioritized
        self.alpha = alpha
        
        self.buffer = deque(maxlen=capacity)
        
        if prioritized:
            self.priorities = deque(maxlen=capacity)
        
        self.position = 0
    
    def push(self, state, action, reward, next_state, done, priority=None):
        """
        Añade una experiencia al buffer
        
        Args:
            state: estado actual
            action: acción tomada
            reward: recompensa obtenida
            next_state: siguiente estado
            done: si el episodio terminó
            priority: prioridad de la experiencia (para prioritized replay)
        """
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            if self.prioritized:
                max_priority = max(self.priorities) if self.priorities else 1.0
                self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = experience
            if self.prioritized:
                self.priorities[self.position] = max(self.priorities) if self.priorities else 1.0
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        """
        Muestrea un batch de experiencias
        
        Args:
            batch_size: tamaño del batch
            beta: parámetro de corrección de importancia para prioritized replay
            
        Returns:
            batch de experiencias procesadas
        """
        if self.prioritized:
            return self._prioritized_sample(batch_size, beta)
        else:
            return self._uniform_sample(batch_size)
    
    def _uniform_sample(self, batch_size):
        """Muestreo uniforme tradicional"""
        experiences = random.sample(self.buffer, batch_size)
        return self._process_batch(experiences)
    
    def _prioritized_sample(self, batch_size, beta):
        """Muestreo priorizado basado en TD error"""
        # Convertir prioridades a probabilidades
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Muestrear índices basado en probabilidades
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calcular pesos de importancia
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalizar
        
        batch = self._process_batch(experiences)
        batch['weights'] = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
        batch['indices'] = indices
        
        return batch
    
    def _process_batch(self, experiences):
        """Procesa un batch de experiencias en tensores"""
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device).unsqueeze(1)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def update_priorities(self, indices, priorities):
        """Actualiza prioridades para prioritized replay"""
        if self.prioritized:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

class FrameStack:
    """
    Utility para apilar frames consecutivos y dar contexto temporal
    """
    
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, frame):
        """Inicializa con el primer frame de un episodio"""
        # Llenar con el mismo frame inicial
        for _ in range(self.num_frames):
            self.frames.append(frame)
        return self.get_state()
    
    def step(self, frame):
        """Añade un nuevo frame"""
        self.frames.append(frame)
        return self.get_state()
    
    def get_state(self):
        """Retorna el estado actual (frames apilados)"""
        return np.stack(self.frames, axis=0)

class EpsilonScheduler:
    """
    Scheduler para epsilon (exploración vs explotación)
    """
    
    def __init__(self, start=1.0, end=0.01, decay_steps=100000):
        """
        Args:
            start: epsilon inicial
            end: epsilon final
            decay_steps: pasos para llegar de start a end
        """
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.current_step = 0
    
    def get_epsilon(self):
        """Calcula epsilon actual"""
        if self.current_step >= self.decay_steps:
            return self.end
        
        # Decaimiento lineal
        epsilon = self.start - (self.start - self.end) * (self.current_step / self.decay_steps)
        return max(epsilon, self.end)
    
    def step(self):
        """Incrementa el paso"""
        self.current_step += 1
    
    def reset(self):
        """Reinicia el scheduler"""
        self.current_step = 0

class RewardShaper:
    """
    Utility para shaping de recompensas (mejorar señal de aprendizaje)
    """
    
    def __init__(self):
        self.last_lives = None
        self.last_score = 0
    
    def shape_reward(self, reward, info, done):
        """
        Mejora las recompensas para facilitar el aprendizaje
        
        Args:
            reward: recompensa original del entorno
            info: información adicional del entorno
            done: si el episodio terminó
            
        Returns:
            recompensa modificada
        """
        shaped_reward = reward
        
        # Recompensa por sobrevivir (pequeña)
        if not done:
            shaped_reward += 0.1
        
        # Penalización por morir
        if done and reward <= 0:
            shaped_reward -= 10
        
        # Bonus por incremento en score
        if 'ale.lives' in info:
            # Penalización por perder vida
            if self.last_lives is not None and info['ale.lives'] < self.last_lives:
                shaped_reward -= 50
            self.last_lives = info['ale.lives']
        
        # Bonus por ganar puntos
        if hasattr(info, 'score') and info.score > self.last_score:
            shaped_reward += (info.score - self.last_score) * 0.1
            self.last_score = info.score
        
        return shaped_reward
    
    def reset(self):
        """Reinicia el shaper para nuevo episodio"""
        self.last_lives = None
        self.last_score = 0

def preprocess_frame(frame):
    """
    Preprocesa frames del juego para la red neuronal
    
    Args:
        frame: frame original del juego [H, W, C]
        
    Returns:
        frame procesado [84, 84]
    """
    import cv2
    
    # Convertir a escala de grises
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    # Redimensionar a 84x84 (estándar para Atari DQN)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_LINEAR)
    
    # Normalizar a [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized

def save_checkpoint(agent, episode, score, filepath):
    """Guarda checkpoint del entrenamiento"""
    checkpoint = {
        'episode': episode,
        'model_state_dict': agent.q_network.state_dict(),
        'target_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'score': score,
        'epsilon': agent.epsilon_scheduler.get_epsilon()
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(agent, filepath, device):
    """Carga checkpoint del entrenamiento"""
    checkpoint = torch.load(filepath, map_location=device)
    agent.q_network.load_state_dict(checkpoint['model_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['episode'], checkpoint['score']