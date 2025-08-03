import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from collections import deque

# Importar nuestros módulos
from dqn_agent import DQNAgent
from replay_buffer import FrameStack, preprocess_frame
from config import Config, DeviceConfig, print_config, validate_config

def create_environment():
    """Crea y configura el entorno de Pac-Man"""
    env = gym.make('ALE/MsPacman-v5', render_mode=None)
    print(f"🎮 Entorno creado: {env.spec.id}")
    print(f"🎯 Acciones disponibles: {env.action_space.n}")
    print(f"📺 Observación: {env.observation_space.shape}")
    return env

def setup_training(config):
    """Configura el entorno de entrenamiento"""
    # Crear directorios necesarios
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Dispositivo de entrenamiento: {device}")
    
    return device

def train_dqn_agent(config=None, resume_from=None):
    """
    Función principal de entrenamiento del agente DQN
    
    Args:
        config: configuración de hiperparámetros
        resume_from: checkpoint para resumir entrenamiento
    """
    # Configuración por defecto optimizada para CPU
    if config is None:
        config = DeviceConfig.get_config_for_device('cpu')
    
    # Validar configuración
    validate_config(config)
    print_config(config)
    
    # Setup del entrenamiento
    device = setup_training(config)
    
    # Crear entorno
    env = create_environment()
    n_actions = env.action_space.n
    
    # Crear agente
    agent = DQNAgent(
        state_shape=config.INPUT_SHAPE,
        n_actions=n_actions,
        config=config,
        device=device
    )
    
    # Cargar checkpoint si se especifica
    start_episode = 0
    if resume_from and os.path.exists(resume_from):
        if agent.load_model(resume_from):
            start_episode = agent.episode_count
            print(f"🔄 Resumiendo desde episodio {start_episode}")
    
    # Inicializar métricas
    episode_rewards = []
    episode_lengths = []
    losses = []
    eval_scores = []
    
    # Frame stacking utility
    frame_stack = FrameStack(config.FRAME_STACK)
    
    print(f"\n🚀 Iniciando entrenamiento por {config.TOTAL_EPISODES - start_episode} episodios...")
    
    # Loop principal de entrenamiento
    start_time = time.time()
    
    for episode in tqdm(range(start_episode, config.TOTAL_EPISODES), 
                       desc="Entrenamiento", 
                       unit="episodio"):
        
        # Reiniciar entorno
        raw_state, info = env.reset()
        processed_frame = preprocess_frame(raw_state)
        state = frame_stack.reset(processed_frame)
        
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        agent.reset_episode()
        
        # Loop del episodio
        for step in range(config.MAX_STEPS_PER_EPISODE):
            # Seleccionar acción
            action = agent.act(state, training=True)
            
            # Ejecutar acción en el entorno
            next_raw_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Procesar siguiente estado
            next_processed_frame = preprocess_frame(next_raw_state)
            next_state = frame_stack.step(next_processed_frame)
            
            # Almacenar experiencia
            agent.remember(state, action, reward, next_state, done, info)
            
            # Entrenar el agente
            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)
            
            # Actualizar estado
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # Guardar métricas del episodio
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode_losses:
            losses.extend(episode_losses)
        
        # Logging y guardado periódico
        if episode % config.LOG_FREQ == 0:
            stats = agent.get_stats()
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            
            elapsed_time = time.time() - start_time
            eps_per_minute = (episode - start_episode + 1) / (elapsed_time / 60)
            
            print(f"\n📊 Episodio {episode}")
            print(f"   Recompensa: {episode_reward:.1f} (promedio: {avg_reward:.1f})")
            print(f"   Longitud: {episode_length} (promedio: {avg_length:.1f})")
            print(f"   Epsilon: {stats['epsilon']:.3f}")
            print(f"   Buffer: {stats['buffer_size']:,}/{config.BUFFER_SIZE:,}")
            print(f"   Velocidad: {eps_per_minute:.1f} ep/min")
            
            if episode_losses:
                print(f"   Pérdida promedio: {np.mean(episode_losses):.4f}")
        
        # Guardar modelo periódicamente
        if episode % config.SAVE_FREQ == 0 and episode > 0:
            model_path = os.path.join(config.MODEL_DIR, f'pacman_dqn_episode_{episode}.pth')
            agent.save_model(model_path)
        
        # Evaluación periódica
        if episode % config.EVAL_FREQ == 0 and episode > 0:
            eval_score = evaluate_agent(agent, env, config, episodes=config.EVAL_EPISODES)
            eval_scores.append((episode, eval_score))
            print(f"🎯 Evaluación episodio {episode}: {eval_score:.1f} puntos promedio")
    
    # Guardar modelo final
    final_model_path = os.path.join(config.MODEL_DIR, 'pacman_dqn_final.pth')
    agent.save_model(final_model_path)
    
    # Cerrar entorno
    env.close()
    
    # Crear gráficas de entrenamiento
    plot_training_results(episode_rewards, losses, eval_scores, config)
    
    print(f"\n🎉 Entrenamiento completado!")
    print(f"⏱️  Tiempo total: {(time.time() - start_time)/60:.1f} minutos")
    print(f"💾 Modelo final guardado en: {final_model_path}")
    
    return agent, episode_rewards, losses

def evaluate_agent(agent, env, config, episodes=5):
    """
    Evalúa el agente entrenado sin exploración
    
    Args:
        agent: agente a evaluar
        env: entorno de evaluación
        config: configuración
        episodes: número de episodios de evaluación
        
    Returns:
        puntuación promedio
    """
    agent.set_eval_mode()
    frame_stack = FrameStack(config.FRAME_STACK)
    
    total_rewards = []
    
    for _ in range(episodes):
        raw_state, _ = env.reset()
        processed_frame = preprocess_frame(raw_state)
        state = frame_stack.reset(processed_frame)
        
        episode_reward = 0
        done = False
        
        while not done:
            # Acción sin exploración (epsilon = 0)
            action = agent.act(state, training=False)
            
            next_raw_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_processed_frame = preprocess_frame(next_raw_state)
            state = frame_stack.step(next_processed_frame)
            
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    agent.set_train_mode()
    return np.mean(total_rewards)

def plot_training_results(episode_rewards, losses, eval_scores, config):
    """Crea gráficas del progreso de entrenamiento"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Progreso de Entrenamiento DQN - Pac-Man', fontsize=16)
    
    # Recompensas por episodio
    axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue', label='Por episodio')
    if len(episode_rewards) >= 100:
        smoothed = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        axes[0, 0].plot(range(99, len(episode_rewards)), smoothed, 
                       color='red', linewidth=2, label='Promedio móvil (100)')
    axes[0, 0].set_title('Recompensas por Episodio')
    axes[0, 0].set_xlabel('Episodio')
    axes[0, 0].set_ylabel('Recompensa')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pérdidas de entrenamiento
    if losses:
        axes[0, 1].plot(losses, alpha=0.6, color='orange')
        axes[0, 1].set_title('Pérdida de Entrenamiento')
        axes[0, 1].set_xlabel('Paso de entrenamiento')
        axes[0, 1].set_ylabel('Pérdida MSE')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Evaluaciones periódicas
    if eval_scores:
        episodes, scores = zip(*eval_scores)
        axes[1, 0].plot(episodes, scores, 'go-', linewidth=2, markersize=6)
        axes[1, 0].set_title('Evaluaciones Periódicas')
        axes[1, 0].set_xlabel('Episodio')
        axes[1, 0].set_ylabel('Puntuación Promedio')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Distribución de recompensas
    axes[1, 1].hist(episode_rewards, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                      linewidth=2, label=f'Media: {np.mean(episode_rewards):.1f}')
    axes[1, 1].set_title('Distribución de Recompensas')
    axes[1, 1].set_xlabel('Recompensa')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar gráficas
    plot_path = os.path.join(config.LOG_DIR, 'training_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Gráficas guardadas en: {plot_path}")

def main():
    """Función principal"""
    print("🎮 ENTRENAMIENTO DQN PARA PAC-MAN")
    print("="*50)
    
    # Configuración optimizada para CPU
    config = DeviceConfig.get_config_for_device('cpu')
    
    # Para pruebas rápidas, descomenta la siguiente línea:
    # config = Config.quick_test()
    
    # Iniciar entrenamiento
    agent, rewards, losses = train_dqn_agent(config)
    
    print("\n🎯 Para probar el agente entrenado, ejecuta:")
    print("python test_agent.py")

if __name__ == "__main__":
    main()