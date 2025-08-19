"""
Script para continuar entrenamiento desde un modelo guardado
"""

from train_pacman import train_dqn_agent
from config import Config, DeviceConfig
import os

def continue_training_from_checkpoint():
    """Continúa entrenamiento desde el último modelo guardado"""
    
    # Configuración mejorada para más episodios
    config = DeviceConfig.get_config_for_device('cpu')
    
    # CONFIGURACIÓN MEJORADA PARA MÁS ENTRENAMIENTO
    config.TOTAL_EPISODES = 2000  # Más episodios
    config.LEARNING_RATE = 5e-5   # Learning rate más bajo para fine-tuning
    config.EPSILON_START = 0.1    # Menos exploración inicial (ya aprendió algo)
    config.EPSILON_END = 0.001    # Exploración mínima más baja
    config.EPSILON_DECAY_STEPS = 200000  # Más pasos para decaimiento
    
    # Buscar el modelo más reciente
    model_dir = "models"
    checkpoint = None
    
    if os.path.exists(os.path.join(model_dir, "pacman_dqn_final.pth")):
        checkpoint = os.path.join(model_dir, "pacman_dqn_final.pth")
        print("🔄 Continuando desde modelo final")
    else:
        # Buscar el episodio más alto
        model_files = [f for f in os.listdir(model_dir) if f.startswith("pacman_dqn_episode_")]
        if model_files:
            episodes = [int(f.split("_")[3].split(".")[0]) for f in model_files]
            latest_episode = max(episodes)
            checkpoint = os.path.join(model_dir, f"pacman_dqn_episode_{latest_episode}.pth")
            print(f"🔄 Continuando desde episodio {latest_episode}")
    
    if not checkpoint:
        print("❌ No se encontró modelo previo. Iniciando desde cero...")
        checkpoint = None
    
    print(f"📈 Entrenamiento extendido: {config.TOTAL_EPISODES} episodios totales")
    
    # Iniciar entrenamiento continuado
    agent, rewards, losses = train_dqn_agent(config=config, resume_from=checkpoint)
    
    return agent, rewards, losses

if __name__ == "__main__":
    continue_training_from_checkpoint()