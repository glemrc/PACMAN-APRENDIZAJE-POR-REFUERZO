"""
Configuración de hiperparámetros para el entrenamiento de DQN en Pac-Man
"""

class Config:
    """Configuración centralizada del proyecto"""
    
    # ===============================
    # CONFIGURACIÓN DEL ENTORNO
    # ===============================
    ENV_NAME = 'ALE/MsPacman-v5'
    FRAME_STACK = 4  # Número de frames consecutivos a apilar
    FRAME_SIZE = 84  # Tamaño de frames después del preprocesamiento
    SKIP_FRAMES = 4  # Frames a saltar entre acciones (para velocidad)
    
    # ===============================
    # ARQUITECTURA DE LA RED
    # ===============================
    INPUT_SHAPE = (FRAME_STACK, FRAME_SIZE, FRAME_SIZE)  # (4, 84, 84)
    HIDDEN_SIZE = 512
    LEARNING_RATE = 1e-4
    
    # ===============================
    # PARÁMETROS DE DQN
    # ===============================
    GAMMA = 0.99  # Factor de descuento
    BATCH_SIZE = 32  # Tamaño del batch para entrenamiento
    BUFFER_SIZE = 100000  # Tamaño del buffer de replay
    
    # Actualización de red objetivo
    TARGET_UPDATE_FREQ = 1000  # Cada cuántos pasos actualizar target network
    SOFT_UPDATE = False  # Usar actualización suave vs dura
    TAU = 0.001  # Factor para actualización suave
    
    # ===============================
    # ESTRATEGIA DE EXPLORACIÓN
    # ===============================
    EPSILON_START = 1.0  # Exploración inicial (100%)
    EPSILON_END = 0.01   # Exploración mínima (1%)
    EPSILON_DECAY_STEPS = 100000  # Pasos para decaimiento completo
    
    # ===============================
    # ENTRENAMIENTO
    # ===============================
    TOTAL_EPISODES = 1000  # Número total de episodios
    MAX_STEPS_PER_EPISODE = 10000  # Máximo de pasos por episodio
    
    # Cuándo empezar a entrenar
    LEARNING_STARTS = 10000  # Pasos antes de empezar entrenamiento
    TRAIN_FREQ = 4  # Entrenar cada N pasos
    
    # ===============================
    # LOGGING Y GUARDADO
    # ===============================
    LOG_FREQ = 50  # Cada cuántos episodios mostrar progreso
    SAVE_FREQ = 100  # Cada cuántos episodios guardar modelo
    EVAL_FREQ = 100  # Cada cuántos episodios evaluar
    EVAL_EPISODES = 5  # Episodios para evaluación
    
    # Directorios
    MODEL_DIR = 'models'
    LOG_DIR = 'logs'
    CHECKPOINT_DIR = 'checkpoints'
    
    # ===============================
    # MEJORAS OPCIONALES
    # ===============================
    USE_PRIORITIZED_REPLAY = False  # Usar prioritized experience replay
    USE_DOUBLE_DQN = True  # Usar Double DQN
    USE_DUELING_DQN = False  # Usar Dueling DQN (más complejo)
    USE_NOISY_NETWORKS = False  # Usar redes ruidosas para exploración
    
    # ===============================
    # REWARD SHAPING
    # ===============================
    USE_REWARD_SHAPING = True  # Mejorar señales de recompensa
    SURVIVAL_REWARD = 0.01  # Pequeña recompensa por sobrevivir
    DEATH_PENALTY = -10.0  # Penalización por morir
    LIFE_PENALTY = -50.0  # Penalización por perder vida
    
    # ===============================
    # CONFIGURACIONES RÁPIDAS PREDEFINIDAS
    # ===============================
    @classmethod
    def quick_test(cls):
        """Configuración para pruebas rápidas (menor calidad)"""
        cls.TOTAL_EPISODES = 100
        cls.BUFFER_SIZE = 10000
        cls.LEARNING_STARTS = 1000
        cls.EPSILON_DECAY_STEPS = 10000
        cls.TARGET_UPDATE_FREQ = 100
        cls.LOG_FREQ = 10
        cls.SAVE_FREQ = 25
        cls.EVAL_FREQ = 25
        return cls
    
    @classmethod
    def high_quality(cls):
        """Configuración para entrenamiento de alta calidad"""
        cls.TOTAL_EPISODES = 2000
        cls.BUFFER_SIZE = 200000
        cls.LEARNING_STARTS = 50000
        cls.EPSILON_DECAY_STEPS = 500000
        cls.TARGET_UPDATE_FREQ = 2000
        cls.BATCH_SIZE = 64
        cls.USE_PRIORITIZED_REPLAY = True
        return cls
    
    @classmethod
    def cpu_optimized(cls):
        """Configuración optimizada para CPU (tu caso)"""
        cls.BATCH_SIZE = 16  # Batch más pequeño para CPU
        cls.BUFFER_SIZE = 50000  # Buffer más pequeño
        cls.LEARNING_STARTS = 5000  # Empezar antes
        cls.TARGET_UPDATE_FREQ = 500  # Actualizar más frecuentemente
        cls.TOTAL_EPISODES = 500  # Menos episodios inicialmente
        return cls
    
    @classmethod
    def debug_mode(cls):
        """Configuración para debugging"""
        cls.TOTAL_EPISODES = 10
        cls.MAX_STEPS_PER_EPISODE = 1000
        cls.LEARNING_STARTS = 100
        cls.LOG_FREQ = 1
        cls.SAVE_FREQ = 5
        cls.BUFFER_SIZE = 1000
        cls.EPSILON_DECAY_STEPS = 1000
        return cls

# Configuraciones por dispositivo
class DeviceConfig:
    """Configuraciones específicas según el hardware disponible"""
    
    @staticmethod
    def get_config_for_device(device_type):
        """
        Retorna configuración optimizada según el dispositivo
        
        Args:
            device_type: 'cpu', 'cuda', 'mps' (Mac M1/M2)
        """
        if device_type == 'cpu':
            return Config.cpu_optimized()
        elif device_type == 'cuda':
            return Config.high_quality()
        elif device_type == 'mps':  # Mac M1/M2
            config = Config.cpu_optimized()
            config.BATCH_SIZE = 32  # M1/M2 pueden manejar batch más grandes
            return config
        else:
            return Config()

# Validación de configuración
def validate_config(config):
    """Valida que la configuración sea consistente"""
    assert config.BATCH_SIZE <= config.BUFFER_SIZE, "Batch size debe ser menor que buffer size"
    assert config.LEARNING_STARTS >= config.BATCH_SIZE, "Learning starts debe ser >= batch size"
    assert config.EPSILON_END < config.EPSILON_START, "Epsilon end debe ser menor que epsilon start"
    assert config.GAMMA > 0 and config.GAMMA <= 1, "Gamma debe estar entre 0 y 1"
    
    print("✅ Configuración validada correctamente")
    return True

# Función para mostrar configuración actual
def print_config(config):
    """Imprime la configuración actual de manera legible"""
    print("\n" + "="*50)
    print("🎮 CONFIGURACIÓN DE ENTRENAMIENTO DQN PAC-MAN")
    print("="*50)
    
    print(f"🎯 Episodios totales: {config.TOTAL_EPISODES}")
    print(f"🧠 Tamaño de batch: {config.BATCH_SIZE}")
    print(f"💾 Tamaño de buffer: {config.BUFFER_SIZE:,}")
    print(f"📏 Learning rate: {config.LEARNING_RATE}")
    print(f"🎲 Epsilon: {config.EPSILON_START} → {config.EPSILON_END}")
    print(f"🎯 Gamma (descuento): {config.GAMMA}")
    print(f"🔄 Target update freq: {config.TARGET_UPDATE_FREQ}")
    print(f"📊 Log cada: {config.LOG_FREQ} episodios")
    print(f"💾 Guardar cada: {config.SAVE_FREQ} episodios")
    
    mejoras = []
    if config.USE_DOUBLE_DQN: mejoras.append("Double DQN")
    if config.USE_PRIORITIZED_REPLAY: mejoras.append("Prioritized Replay")
    if config.USE_DUELING_DQN: mejoras.append("Dueling DQN")
    if config.USE_REWARD_SHAPING: mejoras.append("Reward Shaping")
    
    if mejoras:
        print(f"🚀 Mejoras activas: {', '.join(mejoras)}")
    
    print("="*50 + "\n")

# Configuración por defecto basada en tu hardware (CPU)
DEFAULT_CONFIG = DeviceConfig.get_config_for_device('cpu')