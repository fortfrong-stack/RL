"""
Serialization manager for complex objects with improved handling.
"""

import os
import pickle
import json
import torch
import numpy as np
from datetime import datetime
import hashlib


class SerializationManager:
    """Manager for serializing and deserializing complex RL objects."""
    
    @staticmethod
    def save_model(agent, filepath, metadata=None):
        """
        Save an RL agent model with enhanced metadata and validation.
        
        Args:
            agent: The RL agent to save
            filepath: Path to save the model
            metadata: Additional metadata to save with the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare metadata
        model_metadata = {
            'timestamp': datetime.now().isoformat(),
            'algorithm': getattr(agent, '__class__.__name__', 'Unknown'),
            'input_size': getattr(agent, 'input_size', None),
            'output_size': getattr(agent, 'output_size', None),
            'framework': 'torch' if torch.__version__ else 'unknown',
            'torch_version': torch.__version__ if torch.__version__ else None,
            'device': str(getattr(agent, 'device', 'cpu')),
            'custom_metadata': metadata or {}
        }
        
        # Save the model based on its type
        if hasattr(agent, 'save'):
            # Use the agent's own save method if available
            agent.save(filepath)
            
            # Add our metadata separately
            meta_filepath = filepath.replace('.pth', '_meta.json').replace('.pt', '_meta.json')
            with open(meta_filepath, 'w') as f:
                json.dump(model_metadata, f, indent=2)
        else:
            # Generic save for agents without their own save method
            model_data = {
                'model_weights': SerializationManager._extract_weights(agent),
                'metadata': model_metadata
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
    
    @staticmethod
    def load_model(agent_class, filepath, device=None):
        """
        Load an RL agent model with validation and error handling.
        
        Args:
            agent_class: The class of the agent to instantiate
            filepath: Path to load the model from
            device: Device to load the model onto (optional)
            
        Returns:
            Loaded agent instance
        """
        # Check if metadata file exists
        meta_filepath = filepath.replace('.pth', '_meta.json').replace('.pt', '_meta.json')
        metadata = None
        
        if os.path.exists(meta_filepath):
            with open(meta_filepath, 'r') as f:
                metadata = json.load(f)
        
        # Load the model based on file extension and agent capabilities
        if hasattr(agent_class, 'load'):
            # Create an instance of the agent first
            # We need to get the input/output sizes somehow - try from metadata or make assumptions
            input_size = metadata['input_size'] if metadata and 'input_size' in metadata else 128
            output_size = metadata['output_size'] if metadata and 'output_size' in metadata else 5
            
            agent = agent_class(input_size=input_size, output_size=output_size)
            agent.load(filepath)
            
            # Set device if specified
            if device:
                agent.device = torch.device(device)
            
            return agent
        else:
            # Generic load for agents without their own load method
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Reconstruct the agent
            agent = agent_class(**model_data.get('init_params', {}))
            SerializationManager._restore_weights(agent, model_data['model_weights'])
            
            return agent
    
    @staticmethod
    def _extract_weights(agent):
        """Extract model weights from an agent."""
        if hasattr(agent, 'q_network') and hasattr(agent.q_network, 'state_dict'):
            # For DQN-like agents
            return {
                'q_network': agent.q_network.state_dict(),
                'target_network': agent.target_network.state_dict() if hasattr(agent, 'target_network') else None,
                'optimizer': agent.optimizer.state_dict() if hasattr(agent, 'optimizer') else None
            }
        elif hasattr(agent, 'actor_network') and hasattr(agent, 'critic_network'):
            # For actor-critic agents like A3C/PPO
            return {
                'actor_network': agent.actor_network.state_dict(),
                'critic_network': agent.critic_network.state_dict(),
                'shared_layers': agent.shared_layers.state_dict() if hasattr(agent, 'shared_layers') else None,
                'actor_optimizer': agent.actor_optimizer.state_dict() if hasattr(agent, 'actor_optimizer') else None,
                'critic_optimizer': agent.critic_optimizer.state_dict() if hasattr(agent, 'critic_optimizer') else None
            }
        else:
            # Generic approach - try to find any torch modules
            weights = {}
            for attr_name in dir(agent):
                attr = getattr(agent, attr_name)
                if hasattr(attr, 'state_dict') and callable(getattr(attr, 'state_dict')):
                    try:
                        weights[attr_name] = attr.state_dict()
                    except:
                        continue  # Skip if state_dict fails
            return weights
    
    @staticmethod
    def _restore_weights(agent, weights):
        """Restore model weights to an agent."""
        if 'q_network' in weights and hasattr(agent, 'q_network'):
            # For DQN-like agents
            agent.q_network.load_state_dict(weights['q_network'])
            if weights['target_network'] and hasattr(agent, 'target_network'):
                agent.target_network.load_state_dict(weights['target_network'])
            if weights['optimizer'] and hasattr(agent, 'optimizer'):
                agent.optimizer.load_state_dict(weights['optimizer'])
        elif 'actor_network' in weights and hasattr(agent, 'actor_network'):
            # For actor-critic agents
            agent.actor_network.load_state_dict(weights['actor_network'])
            agent.critic_network.load_state_dict(weights['critic_network'])
            if weights['shared_layers'] and hasattr(agent, 'shared_layers'):
                agent.shared_layers.load_state_dict(weights['shared_layers'])
            if weights['actor_optimizer'] and hasattr(agent, 'actor_optimizer'):
                agent.actor_optimizer.load_state_dict(weights['actor_optimizer'])
            if weights['critic_optimizer'] and hasattr(agent, 'critic_optimizer'):
                agent.critic_optimizer.load_state_dict(weights['critic_optimizer'])
        else:
            # Generic approach
            for attr_name, weight_dict in weights.items():
                if hasattr(agent, attr_name):
                    attr = getattr(agent, attr_name)
                    if hasattr(attr, 'load_state_dict') and callable(getattr(attr, 'load_state_dict')):
                        try:
                            attr.load_state_dict(weight_dict)
                        except:
                            continue  # Skip if load_state_dict fails
    
    @staticmethod
    def validate_model(filepath):
        """Validate that a saved model file is not corrupted."""
        if not os.path.exists(filepath):
            return False, "File does not exist"
        
        # Check file size
        if os.path.getsize(filepath) == 0:
            return False, "File is empty"
        
        # Try to load the file to check if it's valid
        try:
            if filepath.endswith(('.pth', '.pt')):
                # PyTorch model
                checkpoint = torch.load(filepath, map_location='cpu')
                return True, "Valid PyTorch model file"
            elif filepath.endswith('.pkl'):
                # Pickle file
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                return True, "Valid pickle file"
            else:
                # Try both approaches
                try:
                    checkpoint = torch.load(filepath, map_location='cpu')
                    return True, "Valid PyTorch model file"
                except:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                    return True, "Valid pickle file"
        except Exception as e:
            return False, f"Invalid file format: {str(e)}"
    
    @staticmethod
    def backup_model(filepath, backup_dir="backups"):
        """Create a backup of a model file."""
        if not os.path.exists(filepath):
            return False
        
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        # Generate backup filename with timestamp
        base_name = os.path.basename(filepath)
        name, ext = os.path.splitext(base_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{name}_backup_{timestamp}{ext}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Copy the file
        with open(filepath, 'rb') as src, open(backup_path, 'wb') as dst:
            dst.write(src.read())
        
        return backup_path
    
    @staticmethod
    def get_model_info(filepath):
        """Get information about a saved model."""
        if not os.path.exists(filepath):
            return None
        
        try:
            if filepath.endswith(('.pth', '.pt')):
                # PyTorch model
                checkpoint = torch.load(filepath, map_location='cpu')
                
                info = {
                    'file_path': filepath,
                    'file_size': os.path.getsize(filepath),
                    'file_format': 'PyTorch',
                    'keys': list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dictionary',
                    'timestamp': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                }
                
                # Add specific information based on keys
                if isinstance(checkpoint, dict):
                    if 'q_network_state_dict' in checkpoint:
                        info['algorithm'] = 'DQN'
                    elif 'actor_network_state_dict' in checkpoint:
                        info['algorithm'] = 'PPO'
                    elif 'shared_layers_state_dict' in checkpoint:
                        info['algorithm'] = 'A3C'
                    else:
                        info['algorithm'] = 'Unknown'
                
                return info
            else:
                # Other formats
                return {
                    'file_path': filepath,
                    'file_size': os.path.getsize(filepath),
                    'file_format': 'Unknown',
                    'timestamp': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def save_training_state(agent, optimizer, epoch, loss, filepath, extra_data=None):
        """Save a complete training state including model, optimizer, and metadata."""
        training_state = {
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': agent.q_network.state_dict() if hasattr(agent, 'q_network') else agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'timestamp': datetime.now().isoformat(),
            'extra_data': extra_data or {}
        }
        
        torch.save(training_state, filepath)
    
    @staticmethod
    def load_training_state(agent, optimizer, filepath):
        """Load a complete training state."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Load model state
        if hasattr(agent, 'q_network'):
            agent.q_network.load_state_dict(checkpoint['model_state_dict'])
        else:
            agent.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', 0),
            'timestamp': checkpoint.get('timestamp', None),
            'extra_data': checkpoint.get('extra_data', {})
        }


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the SerializationManager
    print("Testing SerializationManager...")
    
    # This would work with actual agents:
    # agent = DQNAgent(input_size=10, output_size=5)
    # SerializationManager.save_model(agent, "models/test_model.pth", {"task": "navigation", "env": "grid"})
    # loaded_agent = SerializationManager.load_model(DQNAgent, "models/test_model.pth")
    
    print("SerializationManager ready for use with RL agents.")