import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQNNetwork(nn.Module):
    """
    Red Neuronal Convolucional para Deep Q-Learning
    
    Arquitectura inspirada en el paper original de DeepMind:
    - 3 capas convolucionales para extracción de características
    - 2 capas completamente conectadas para predicción de valores Q
    """
    
    def __init__(self, input_shape, n_actions, learning_rate=1e-4):
        super(DQNNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        # Capas convolucionales
        # Conv1: 32 filtros, kernel 8x8, stride 4
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization para estabilidad
        
        # Conv2: 64 filtros, kernel 4x4, stride 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv3: 64 filtros, kernel 3x3, stride 1
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Calcular tamaño para capas densas
        conv_out_size = self._get_conv_out(input_shape)
        
        # Capas completamente conectadas
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.dropout1 = nn.Dropout(0.2)  # Dropout para prevenir overfitting
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        
        # Capa de salida - valores Q para cada acción
        self.fc3 = nn.Linear(256, n_actions)
        
        # Inicialización de pesos mejorada
        self._initialize_weights()
    
    def _get_conv_out(self, shape):
        """
        Calcula el tamaño de salida de las capas convolucionales
        pasando un tensor dummy por las capas conv
        """
        o = torch.zeros(1, *shape)
        o = self._conv_forward(o)
        return int(np.prod(o.size()))
    
    def _conv_forward(self, x):
        """Forward pass solo de capas convolucionales"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x
    
    def _initialize_weights(self):
        """Inicialización mejorada de pesos usando Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Propagación hacia adelante
        
        Args:
            x: tensor de entrada [batch_size, channels, height, width]
            
        Returns:
            Q-valores para cada acción [batch_size, n_actions]
        """
        # Normalizar entrada si es necesario
        if x.max() > 1.0:
            x = x / 255.0
        
        # Capas convolucionales
        x = self._conv_forward(x)
        
        # Aplanar para capas densas
        x = x.view(x.size(0), -1)
        
        # Capas completamente conectadas
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Salida final - valores Q
        q_values = self.fc3(x)
        
        return q_values
    
    def get_action(self, state, epsilon=0.0):
        """
        Selecciona acción usando política epsilon-greedy
        
        Args:
            state: estado actual del juego
            epsilon: probabilidad de acción aleatoria
            
        Returns:
            acción seleccionada
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        
        with torch.no_grad():
            if len(state.shape) == 3:  # Añadir dimensión batch si es necesario
                state = state.unsqueeze(0)
            
            q_values = self.forward(state)
            return q_values.argmax().item()

# Funciones auxiliares para la red
def create_dqn_networks(input_shape, n_actions, device, learning_rate=1e-4):
    """
    Crea las redes principal y objetivo para DQN
    
    Returns:
        q_network: red principal para entrenamiento
        target_network: red objetivo para estabilidad
    """
    q_network = DQNNetwork(input_shape, n_actions, learning_rate).to(device)
    target_network = DQNNetwork(input_shape, n_actions, learning_rate).to(device)
    
    # Copiar pesos iniciales
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()  # Modo evaluación para red objetivo
    
    return q_network, target_network

def soft_update_target_network(q_network, target_network, tau=0.001):
    """
    Actualización suave de la red objetivo (en lugar de copia completa)
    
    Args:
        tau: factor de actualización (0.001 = actualización muy suave)
    """
    for target_param, local_param in zip(target_network.parameters(), q_network.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def hard_update_target_network(q_network, target_network):
    """Actualización completa de la red objetivo"""
    target_network.load_state_dict(q_network.state_dict())