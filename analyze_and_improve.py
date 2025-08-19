"""
Análisis del entrenamiento actual y sugerencias de mejora
"""

import torch
import config
torch.serialization.add_safe_globals([config.Config])
import numpy as np
import matplotlib.pyplot as plt
from test_agent import benchmark_agent
import os

def analyze_current_model():
    """Analiza el rendimiento del modelo actual"""
    
    print("🔍 ANÁLISIS DEL MODELO ACTUAL")
    print("="*40)
    
    # Verificar si existe el modelo
    model_path = "models/pacman_dqn_final.pth"
    if not os.path.exists(model_path):
        print("❌ No se encontró modelo final")
        return None
    
    # Cargar y analizar checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"📊 Información del modelo:")
    print(f"   Episodio: {checkpoint.get('episode_count', 'N/A')}")
    print(f"   Pasos totales: {checkpoint.get('step_count', 'N/A')}")
    print(f"   Epsilon final: {checkpoint.get('epsilon', 'N/A'):.4f}")
    
    # Analizar historial de recompensas
    episode_rewards = checkpoint.get('episode_rewards', [])
    if episode_rewards:
        recent_performance = np.mean(episode_rewards[-100:])
        best_performance = np.max(episode_rewards)
        worst_performance = np.min(episode_rewards)
        
        print(f"   Rendimiento promedio (últimos 100): {recent_performance:.1f}")
        print(f"   Mejor episodio: {best_performance:.1f}")
        print(f"   Peor episodio: {worst_performance:.1f}")
        print(f"   Desviación estándar: {np.std(episode_rewards[-100:]):.1f}")
        
        # Análisis de tendencia
        if len(episode_rewards) >= 200:
            first_half = np.mean(episode_rewards[:len(episode_rewards)//2])
            second_half = np.mean(episode_rewards[len(episode_rewards)//2:])
            improvement = second_half - first_half
            
            print(f"   Mejora durante entrenamiento: {improvement:.1f} puntos")
            
            if improvement > 100:
                print("✅ El modelo está aprendiendo bien")
            elif improvement > 0:
                print("⚠️  Mejora lenta - considerar más entrenamiento")
            else:
                print("❌ Sin mejora - revisar hiperparámetros")
    
    # Benchmark detallado
    print("\n🎯 Ejecutando benchmark...")
    scores, lengths = benchmark_agent(model_path, episodes=20)
    
    return {
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': np.max(scores),
        'consistency': np.std(scores) / np.mean(scores)  # Menor es mejor
    }

def suggest_improvements(analysis):
    """Sugiere mejoras basadas en el análisis"""
    
    print("\n💡 SUGERENCIAS DE MEJORA")
    print("="*30)
    
    if analysis is None:
        print("❌ No se pudo analizar el modelo")
        return
    
    avg_score = analysis['avg_score']
    consistency = analysis['consistency']
    
    # Sugerencias basadas en rendimiento
    if avg_score < 300:
        print("🔄 RENDIMIENTO BAJO (<300 puntos)")
        print("   Recomendaciones:")
        print("   - Más episodios de entrenamiento (2000-5000)")
        print("   - Activar Prioritized Experience Replay")
        print("   - Reward shaping más agresivo")
        print("   - Learning rate más bajo (1e-5)")
        
    elif avg_score < 800:
        print("📈 RENDIMIENTO MEDIO (300-800 puntos)")
        print("   Recomendaciones:")
        print("   - Continuar entrenamiento (1000+ episodios más)")
        print("   - Buffer más grande (200k experiencias)")
        print("   - Fine-tuning con learning rate bajo")
        
    elif avg_score < 1500:
        print("🎯 BUEN RENDIMIENTO (800-1500 puntos)")
        print("   Recomendaciones:")
        print("   - Fine-tuning cuidadoso")
        print("   - Exploración mínima (epsilon < 0.001)")
        print("   - Evaluación más larga para confirmar")
        
    else:
        print("🏆 EXCELENTE RENDIMIENTO (>1500 puntos)")
        print("   ¡El modelo está listo para uso en robot!")
    
    # Sugerencias basadas en consistencia
    if consistency > 1.0:
        print("\n⚠️  ALTA VARIABILIDAD")
        print("   - El agente es inconsistente")
        print("   - Más entrenamiento para estabilizar")
        print("   - Considerar ensemble de modelos")
    elif consistency < 0.3:
        print("\n✅ BUENA CONSISTENCIA")
        print("   - El agente es predecible")
        print("   - Listo para aplicación robótica")

def create_improvement_plan():
    """Crea un plan de mejora personalizado"""
    
    print("\n📋 PLAN DE MEJORA PERSONALIZADO")
    print("="*35)
    
    analysis = analyze_current_model()
    suggest_improvements(analysis)
    
    print("\n🚀 OPCIONES DE ENTRENAMIENTO:")
    print("1. Continuar entrenamiento actual (+1000 episodios)")
    print("2. Entrenamiento de alto rendimiento (+2000 episodios)")
    print("3. Fine-tuning cuidadoso (+500 episodios, LR bajo)")
    print("4. Entrenamiento desde cero con mejores parámetros")
    
    return analysis

def plot_learning_curve():
    """Grafica la curva de aprendizaje del modelo actual"""
    
    model_path = "models/pacman_dqn_final.pth"
    if not os.path.exists(model_path):
        print("❌ No se encontró modelo para analizar")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu')
    rewards = checkpoint.get('episode_rewards', [])
    losses = checkpoint.get('losses', [])
    
    if not rewards:
        print("❌ No hay datos de entrenamiento en el modelo")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Análisis Detallado del Entrenamiento', fontsize=16)
    
    # Recompensas con tendencia
    axes[0,0].plot(rewards, alpha=0.3, color='blue')
    if len(rewards) >= 50:
        smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
        axes[0,0].plot(range(49, len(rewards)), smoothed, 'r-', linewidth=2)
    axes[0,0].set_title('Progreso de Recompensas')
    axes[0,0].set_xlabel('Episodio')
    axes[0,0].set_ylabel('Recompensa')
    axes[0,0].grid(True, alpha=0.3)
    
    # Distribución de recompensas por fases
    if len(rewards) >= 300:
        early = rewards[:len(rewards)//3]
        mid = rewards[len(rewards)//3:2*len(rewards)//3]
        late = rewards[2*len(rewards)//3:]
        
        axes[0,1].hist([early, mid, late], bins=20, alpha=0.7, 
                      label=['Inicial', 'Medio', 'Final'], color=['red', 'orange', 'green'])
        axes[0,1].set_title('Distribución por Fases')
        axes[0,1].legend()
    
    # Pérdidas si están disponibles
    if losses:
        axes[1,0].plot(losses, alpha=0.6, color='orange')
        axes[1,0].set_title('Pérdida de Entrenamiento')
        axes[1,0].set_yscale('log')
        axes[1,0].grid(True, alpha=0.3)
    
    # Estadísticas móviles
    if len(rewards) >= 100:
        window_means = [np.mean(rewards[i:i+100]) for i in range(len(rewards)-99)]
        window_stds = [np.std(rewards[i:i+100]) for i in range(len(rewards)-99)]
        
        axes[1,1].plot(window_means, label='Media móvil (100 ep)')
        axes[1,1].fill_between(range(len(window_means)), 
                              np.array(window_means) - np.array(window_stds),
                              np.array(window_means) + np.array(window_stds),
                              alpha=0.3)
        axes[1,1].set_title('Estabilidad del Aprendizaje')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Análisis guardado en 'training_analysis.png'")

if __name__ == "__main__":
    # Análisis completo
    create_improvement_plan()
    plot_learning_curve()