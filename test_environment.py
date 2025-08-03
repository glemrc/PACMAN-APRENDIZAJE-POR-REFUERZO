import gymnasium as gym
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def test_environment():
    """Verifica que el entorno de Pac-Man funcione correctamente"""
    print("🔍 Verificando el entorno...")
    
    # Verificar PyTorch y GPU
    print(f"✅ PyTorch versión: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Dispositivo disponible: {device}")
    
    try:
        # Crear entorno de Pac-Man
        env = gym.make('ALE/MsPacman-v5', render_mode=None)
        print("✅ Entorno Pac-Man creado exitosamente")
        
        # Información básica del entorno
        print(f"✅ Número de acciones: {env.action_space.n}")
        print(f"✅ Forma de observación: {env.observation_space.shape}")
        
        # Probar un episodio corto
        state, info = env.reset()
        print(f"✅ Estado inicial - Forma: {state.shape}, Tipo: {state.dtype}")
        
        # Ejecutar algunas acciones aleatorias
        total_reward = 0
        for step in range(100):  # Solo 100 pasos para prueba
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        env.close()
        print(f"✅ Prueba de episodio completada - Recompensa total: {total_reward}")
        
        # Probar preprocesamiento de imagen
        test_frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        processed = preprocess_test_frame(test_frame)
        print(f"✅ Preprocesamiento - Original: {test_frame.shape}, Procesado: {processed.shape}")
        
        print("\n🎉 ¡Todas las verificaciones pasaron exitosamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error durante la verificación: {e}")
        return False

def preprocess_test_frame(frame):
    """Función de prueba para preprocesamiento"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    normalized = resized / 255.0
    return normalized

def show_sample_frames():
    """Muestra algunos frames de ejemplo del juego"""
    print("\n📸 Capturando frames de ejemplo...")
    
    env = gym.make('ALE/MsPacman-v5', render_mode=None)
    state, info = env.reset()
    
    # Capturar varios frames
    frames = []
    for i in range(4):
        if i > 0:
            action = env.action_space.sample()
            state, _, _, _, _ = env.step(action)
        frames.append(state.copy())
    
    env.close()
    
    # Mostrar frames originales y procesados
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('Frames Originales vs Procesados')
    
    for i, frame in enumerate(frames):
        # Frame original
        axes[0, i].imshow(frame)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Frame procesado
        processed = preprocess_test_frame(frame)
        axes[1, i].imshow(processed, cmap='gray')
        axes[1, i].set_title(f'Procesado {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_frames.png')
    plt.show()
    print("✅ Frames guardados en 'sample_frames.png'")

if __name__ == "__main__":
    print("🚀 Iniciando verificación del entorno de Pac-Man...")
    
    if test_environment():
        print("\n¿Quieres ver frames de ejemplo? (s/n):")
        response = input().lower().strip()
        if response == 's' or response == 'si':
            show_sample_frames()
    else:
        print("\n❌ Hay problemas con el entorno. Revisa los errores arriba.")