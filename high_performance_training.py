"""
Configuración de alto rendimiento para entrenar un agente competitivo
"""

from train_pacman import train_dqn_agent
from config import Config
import torch

def create_high_performance_config():
    """Crea configuración optimizada para máximo rendimiento"""
    
    config = Config()
    
    # ===============================
    # CONFIGURACIÓN DE ALTO RENDIMIENTO
    # ===============================
    
    # Más episodios para convergencia completa
    config.TOTAL_EPISODES = 3000
    config.MAX_STEPS_PER_EPISODE = 18000  # Más tiempo por partida
    
    # Buffer más grande para más experiencias diversas
    config.BUFFER_SIZE = 200000  # 4x más grande
    config.LEARNING_STARTS = 20000  # Más experiencias antes de entrenar
    
    # Batch más grande para aprendizaje más estable
    config.BATCH_SIZE = 32  # El doble del anterior
    
    # Learning rate schedule más sofisticado
    config.LEARNING_RATE = 2.5e-4  # Slightly higher inicial
    
    # Exploración más gradual
    config.EPSILON_START = 1.0
    config.EPSILON_END = 0.001  # Exploración mínima muy baja
    config.EPSILON_DECAY_STEPS = 500000  # Decay muy gradual
    
    # Target network update más frecuente
    config.TARGET_UPDATE_FREQ = 2000
    config.TRAIN_FREQ = 4
    
    # Mejoras avanzadas activadas
    config.USE_DOUBLE_DQN = True
    config.USE_PRIORITIZED_REPLAY = True  # ¡ACTIVAR PRIORITIZED REPLAY!
    config.USE_REWARD_SHAPING = True
    
    # Reward shaping más agresivo
    config.SURVIVAL_REWARD = 0.1
    config.DEATH_PENALTY = -50.0
    config.LIFE_PENALTY = -100.0
    
    # Logging más frecuente para monitoreo
    config.LOG_FREQ = 25
    config.SAVE_FREQ = 50
    config.EVAL_FREQ = 50
    config.EVAL_EPISODES = 10  # Más episodios para evaluación
    
    return config

def train_high_performance_agent(resume_from_previous=True):
    """Entrena agente con configuración de alto rendimiento"""
    
    print("🚀 ENTRENAMIENTO DE ALTO RENDIMIENTO")
    print("="*50)
    print("🎯 Objetivo: >2000 puntos promedio")
    print("⏱️  Tiempo estimado: 3-5 horas en CPU")
    print("="*50)
    
    # Crear configuración optimizada
    config = create_high_performance_config()
    
    # Buscar modelo previo si se solicita
    checkpoint = None
    if resume_from_previous:
        import os
        model_dir = "models"
        if os.path.exists(os.path.join(model_dir, "pacman_dqn_final.pth")):
            checkpoint = os.path.join(model_dir, "pacman_dqn_final.pth")
            print("📂 Continuando desde modelo existente")
            
            # Ajustar configuración para continuar
            config.EPSILON_START = 0.05  # Menos exploración al continuar
    
    # Verificar memoria disponible
    print(f"💾 Buffer configurado: {config.BUFFER_SIZE:,} experiencias")
    print(f"🧠 Batch size: {config.BATCH_SIZE}")
    
    # Iniciar entrenamiento
    agent, rewards, losses = train_dqn_agent(
        config=config, 
        resume_from=checkpoint
    )
    
    # Análisis final
    import numpy as np
    final_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    
    print(f"\n🏆 RESULTADO FINAL")
    print(f"📊 Promedio últimos 100 episodios: {final_avg:.1f}")
    print(f"🎯 Mejor episodio: {max(rewards):.1f}")
    
    if final_avg > 1000:
        print("🥇 ¡EXCELENTE! Agente competitivo")
    elif final_avg > 500:
        print("🥈 ¡BIEN! Agente funcional - continuar entrenando")
    else:
        print("🔄 Necesita más entrenamiento o ajuste de hiperparámetros")
    
    return agent, rewards, losses

def quick_performance_boost():
    """Configuración de boost rápido (1000 episodios más)"""
    
    config = Config()
    
    # Configuración de boost
    config.TOTAL_EPISODES = 1500  # 1000 episodios adicionales
    config.BUFFER_SIZE = 100000
    config.BATCH_SIZE = 32
    config.LEARNING_RATE = 1e-4
    
    # Menos exploración (ya tiene experiencia)
    config.EPSILON_START = 0.1
    config.EPSILON_END = 0.001
    config.EPSILON_DECAY_STEPS = 100000
    
    # Mejoras activadas
    config.USE_DOUBLE_DQN = True
    config.USE_PRIORITIZED_REPLAY = True
    config.USE_REWARD_SHAPING = True
    
    return config

# Diferentes opciones de entrenamiento
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("Selecciona modo de entrenamiento:")
        print("1. Alto rendimiento (3000 episodios)")
        print("2. Boost rápido (1000 episodios más)")
        print("3. Continuar entrenamiento actual")
        
        choice = input("Opción (1/2/3): ").strip()
        mode = {"1": "high", "2": "boost", "3": "continue"}[choice]
    
    if mode == "high":
        print("🚀 Modo: ALTO RENDIMIENTO")
        train_high_performance_agent()
    elif mode == "boost":
        print("⚡ Modo: BOOST RÁPIDO")
        config = quick_performance_boost()
        agent, rewards, losses = train_dqn_agent(
            config=config, 
            resume_from="models/pacman_dqn_final.pth"
        )
    elif mode == "continue":
        print("🔄 Modo: CONTINUAR")
        from continue_training import continue_training_from_checkpoint
        continue_training_from_checkpoint()