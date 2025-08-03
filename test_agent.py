import gymnasium as gym
import torch
import numpy as np
import time
import os
from dqn_agent import DQNAgent
from replay_buffer import FrameStack, preprocess_frame
from config import Config, DeviceConfig

def test_trained_agent(model_path=None, episodes=5, render=True, record=False):
    """
    Prueba el agente entrenado
    
    Args:
        model_path: ruta del modelo entrenado
        episodes: número de episodios de prueba
        render: mostrar visualización del juego
        record: grabar video del juego
    """
    print("🎮 PROBANDO AGENTE ENTRENADO")
    print("="*40)
    
    # Configuración
    config = DeviceConfig.get_config_for_device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Buscar modelo si no se especifica
    if model_path is None:
        model_dir = config.MODEL_DIR
        if os.path.exists(os.path.join(model_dir, 'pacman_dqn_final.pth')):
            model_path = os.path.join(model_dir, 'pacman_dqn_final.pth')
        else:
            # Buscar el modelo más reciente
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if model_files:
                model_path = os.path.join(model_dir, sorted(model_files)[-1])
            else:
                print("❌ No se encontró ningún modelo entrenado")
                return
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        return
    
    print(f"📂 Cargando modelo: {model_path}")
    
    # Crear entorno
    render_mode = 'human' if render else None
    if record:
        render_mode = 'rgb_array'
    
    env = gym.make('ALE/MsPacman-v5', render_mode=render_mode)
    n_actions = env.action_space.n
    
    # Crear y cargar agente
    agent = DQNAgent(
        state_shape=config.INPUT_SHAPE,
        n_actions=n_actions,
        config=config,
        device=device
    )
    
    if not agent.load_model(model_path):
        print("❌ Error al cargar el modelo")
        return
    
    # Configurar para evaluación (sin exploración)
    agent.set_eval_mode()
    
    # Frame stacking
    frame_stack = FrameStack(config.FRAME_STACK)
    
    # Métricas de prueba
    test_scores = []
    test_lengths = []
    
    if record:
        frames = []
    
    print(f"\n🎯 Ejecutando {episodes} episodios de prueba...")
    
    for episode in range(episodes):
        raw_state, info = env.reset()
        processed_frame = preprocess_frame(raw_state)
        state = frame_stack.reset(processed_frame)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\n🕹️  Episodio {episode + 1}:")
        
        start_time = time.time()
        
        while not done:
            # Seleccionar acción (sin exploración)
            action = agent.act(state, training=False)
            
            # Ejecutar acción
            next_raw_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Procesar siguiente estado
            next_processed_frame = preprocess_frame(next_raw_state)
            state = frame_stack.step(next_processed_frame)
            
            episode_reward += reward
            episode_length += 1
            
            # Grabar frame si es necesario
            if record:
                frames.append(env.render())
            
            # Mostrar progreso cada cierto tiempo
            if render and episode_length % 100 == 0:
                print(f"   Paso {episode_length}, Puntuación: {episode_reward}")
            
            # Pausa pequeña para visualización
            if render:
                time.sleep(0.02)  # ~50 FPS
        
        episode_time = time.time() - start_time
        
        # Guardar métricas
        test_scores.append(episode_reward)
        test_lengths.append(episode_length)
        
        # Mostrar resultados del episodio
        print(f"   ✅ Completado en {episode_time:.1f}s")
        print(f"   🎯 Puntuación final: {episode_reward}")
        print(f"   📏 Pasos totales: {episode_length}")
        
        # Información adicional si está disponible
        if 'ale.lives' in info:
            print(f"   ❤️  Vidas restantes: {info['ale.lives']}")
    
    env.close()
    
    # Mostrar estadísticas finales
    print("\n📊 ESTADÍSTICAS FINALES")
    print("="*30)
    print(f"Episodios jugados: {episodes}")
    print(f"Puntuación promedio: {np.mean(test_scores):.1f} ± {np.std(test_scores):.1f}")
    print(f"Puntuación máxima: {np.max(test_scores)}")
    print(f"Puntuación mínima: {np.min(test_scores)}")
    print(f"Duración promedio: {np.mean(test_lengths):.1f} pasos")
    
    # Grabar video si se solicitó
    if record and frames:
        save_video(frames, "pacman_gameplay.mp4", fps=25)
    
    return test_scores, test_lengths

def compare_agents(model_paths, episodes=3):
    """
    Compara múltiples modelos entrenados
    
    Args:
        model_paths: lista de rutas de modelos
        episodes: episodios de prueba por modelo
    """
    print("🔄 COMPARANDO AGENTES")
    print("="*30)
    
    results = {}
    
    for i, model_path in enumerate(model_paths):
        print(f"\n🤖 Probando modelo {i+1}: {os.path.basename(model_path)}")
        
        scores, lengths = test_trained_agent(
            model_path=model_path,
            episodes=episodes,
            render=False
        )
        
        results[model_path] = {
            'scores': scores,
            'avg_score': np.mean(scores),
            'avg_length': np.mean(lengths)
        }
    
    # Mostrar comparación
    print("\n📊 COMPARACIÓN DE RESULTADOS")
    print("="*50)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_score'], reverse=True)
    
    for i, (path, result) in enumerate(sorted_results):
        model_name = os.path.basename(path)
        print(f"{i+1:2d}. {model_name:<30} | {result['avg_score']:6.1f} puntos")
    
    return results

def benchmark_agent(model_path=None, episodes=50):
    """
    Realiza un benchmark completo del agente
    
    Args:
        model_path: ruta del modelo
        episodes: número de episodios para benchmark
    """
    print("⚡ BENCHMARK DEL AGENTE")
    print("="*30)
    
    scores, lengths = test_trained_agent(
        model_path=model_path,
        episodes=episodes,
        render=False
    )
    
    # Análisis estadístico
    scores = np.array(scores)
    lengths = np.array(lengths)
    
    print(f"\n📈 ANÁLISIS ESTADÍSTICO ({episodes} episodios)")
    print("="*50)
    print(f"Puntuación media: {np.mean(scores):.2f}")
    print(f"Desviación estándar: {np.std(scores):.2f}")
    print(f"Mediana: {np.median(scores):.2f}")
    print(f"Percentil 25: {np.percentile(scores, 25):.2f}")
    print(f"Percentil 75: {np.percentile(scores, 75):.2f}")
    print(f"Rango: {np.min(scores):.0f} - {np.max(scores):.0f}")
    
    # Clasificación de rendimiento
    if np.mean(scores) > 1000:
        rating = "🥇 EXCELENTE"
    elif np.mean(scores) > 500:
        rating = "🥈 BUENO"
    elif np.mean(scores) > 100:
        rating = "🥉 REGULAR"
    else:
        rating = "❌ NECESITA MEJORA"
    
    print(f"\n🏆 Clasificación: {rating}")
    
    return scores, lengths

def save_video(frames, filename, fps=25):
    """
    Guarda video del gameplay
    
    Args:
        frames: lista de frames
        filename: nombre del archivo
        fps: frames por segundo
    """
    try:
        import cv2
        
        if not frames:
            print("❌ No hay frames para guardar")
            return
        
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convertir RGB a BGR para OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(bgr_frame)
        
        video.release()
        print(f"🎬 Video guardado: {filename}")
        
    except ImportError:
        print("❌ OpenCV no disponible para guardar video")
    except Exception as e:
        print(f"❌ Error al guardar video: {e}")

def main():
    """Función principal del script de testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Probar agente DQN entrenado')
    parser.add_argument('--model', type=str, help='Ruta del modelo a probar')
    parser.add_argument('--episodes', type=int, default=5, help='Número de episodios')
    parser.add_argument('--no-render', action='store_true', help='No mostrar visualización')
    parser.add_argument('--benchmark', action='store_true', help='Ejecutar benchmark')
    parser.add_argument('--record', action='store_true', help='Grabar video')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_agent(args.model, episodes=50)
    else:
        test_trained_agent(
            model_path=args.model,
            episodes=args.episodes,
            render=not args.no_render,
            record=args.record
        )

if __name__ == "__main__":
    main()