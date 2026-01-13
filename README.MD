## **ЭТАП 1: Подготовка базовой инфраструктуры (1-2 дня)**

### Компоненты:
- Python 3.10+
- NumPy (для матричных операций)
- PyTorch (для нейросетей)
- Matplotlib (для визуализации)

### Задачи:
1. **Создать класс `GridWorld`**
   - Инициализация сетки 25×25 как NumPy массив
   - Типы клеток: пустая (0), стена (1), агент (2), источник звука (3)
   - Методы: `reset()`, `get_state()`, `render()`

2. **Реализовать систему координат**
   - Система координат [x, y] где x, y ∈ [0, 24]
   - Методы для проверки границ и перемещения

3. **Создать базовый класс `Agent`**
   - Позиция на сетке
   - Действия: вверх, вниз, влево, вправо, остаться
   - Наблюдения: пока пустой массив

## **ЭТАП 2: Физика звука и стен (2-3 дня)**

### Компоненты:
- NumPy (для вычислений распространения)
- SciPy (для интерполяции, опционально)

### Задачи:
1. **Создать класс `SoundSource`**
   - Параметры:
     - `volume` (0.1-1.0) - определяет радиус распространения
     - `frequency` (0.1-1.0) - вероятность излучения звука за шаг
     - `position` - координаты на сетке
   - Метод `emit_sound()` - генерирует звуковой сигнал с заданной вероятностью

2. **Реализовать распространение звука**
   - Алгоритм волнового распространения (Wavefront propagation):
   ```python
   def propagate_sound(grid, sources, walls):
       sound_map = np.zeros((25, 25))
       for source in sources:
           if random.random() < source.frequency:  # Проверка частоты
               # Распространение от источника с затуханием
               for distance in range(max_distance):
                   for dx, dy in get_neighbors(distance):
                       if in_bounds(x+dx, y+dy):
                           attenuation = source.volume * (0.9 ** distance)
                           sound_map[x+dx][y+dy] += attenuation
       # Учет проницаемости стен
       for wall in walls:
           sound_map[wall.x][wall.y] *= wall.permeability  # 0.25-1.0
       return sound_map
   ```

3. **Создать класс `Wall`**
   - Координаты и проницаемость (permeability: 0.25-1.0)
   - Метод проверки проходимости для агента

## **ЭТАП 3: Система наблюдений и частичной наблюдаемости (2-3 дня)**

### Компоненты:
- PyTorch (для обработки аудио-признаков)
- Librosa (для аудио-анализа)

### Задачи:
1. **Создать аудио-наблюдения для агента**
   - Генерация синтетических аудиосигналов:
     ```python
     def generate_audio_signal(intensity, frequency_content):
         # intensity: сила звука в клетке агента
         # frequency_content: характеристики источника
         duration = 0.1  # секунд
         sample_rate = 44100
         t = np.linspace(0, duration, int(sample_rate * duration))
         
         # Генерация сигнала с заданной частотой
         base_freq = 440 * frequency_content  # базовая частота
         signal = intensity * np.sin(2 * np.pi * base_freq * t)
         
         # Добавление шума и обработка
         noise = np.random.normal(0, 0.01, len(t))
         return signal + noise
     ```

2. **Извлечение признаков из аудио**
   - Реализовать MFCC (Mel-frequency cepstral coefficients) вручную или с librosa
   - Спектрограммы как входные признаки для нейросети

3. **Создать наблюдаемое пространство агента**
   - Агент получает только аудиосигнал из своей текущей клетки
   - Наблюдение = [MFCC коэффициенты, интенсивность, спектральные признаки]

## **ЭТАП 4: Реализация трех задач (3-4 дня)**

### Компоненты:
- Все предыдущие компоненты
- Система ревардов

### Задачи:

#### **Задача 1: Поиск всех источников звука**
- **Параметры**: 1-5 источников
- **Ревард система**:
  - +10 за нахождение нового источника
  - -0.1 за каждый шаг (стимул к эффективности)
  - +50 за нахождение всех источников
- **Состояние завершения**: все источники найдены или макс. шагов

#### **Задача 2: Найти самое тихое место**
- **Параметры**: 1-5 источников
- **Ревард система**:
  - Ревард = -1 × текущая интенсивность звука в клетке
  - +100 за достижение клетки с минимальной интенсивностью
- **Состояние завершения**: агент в самой тихой клетке или макс. шагов

#### **Задача 3: Двигаться за движущимся источником**
- **Параметры**: 1 источник, который двигается рандомно
- **Ревард система**:
  - +5 за уменьшение расстояния до источника
  - -5 за увеличение расстояния
  - +100 за поимку источника (расстояние < 2 клетки)
- **Движение источника**: рандомное перемещение каждые N шагов

## **ЭТАП 5: Система обучения и тестирования (2-3 дня)**

### Компоненты:
- PyTorch (DQN/PPO реализация)
- NumPy (буфер воспроизведения)

### Задачи:
1. **Создать систему рандомной генерации для обучения**
   ```python
   def generate_random_environment(task_type):
       grid = np.zeros((25, 25))
       
       # Случайные стены (20-40% клеток)
       num_walls = random.randint(125, 250)  # 20-40% от 625
       for _ in range(num_walls):
           x, y = random.randint(0, 24), random.randint(0, 24)
           permeability = random.uniform(0.25, 1.0)
           grid[x][y] = 1  # стена
           walls.append(Wall(x, y, permeability))
       
       # Случайные источники в зависимости от задачи
       if task_type in [1, 2]:
           num_sources = random.randint(1, 5)
       else:  # task_type == 3
           num_sources = 1
           
       for _ in range(num_sources):
           x, y = get_random_valid_position(grid)
           volume = random.uniform(0.3, 1.0)
           frequency = random.uniform(0.2, 1.0)
           sources.append(SoundSource(x, y, volume, frequency))
       
       # Случайная позиция агента
       agent_pos = get_random_valid_position(grid)
       
       return Environment(grid, agent_pos, sources, walls, task_type)
   ```

2. **Создать систему ручного размещения для тестирования**
   ```python
   def manual_environment_setup(task_type):
       grid = np.zeros((25, 25))
       print("Ручная настройка среды. Введите координаты:")
       
       # Ввод стен
       num_walls = int(input("Количество стен: "))
       for i in range(num_walls):
           x = int(input(f"Стена {i+1} X (0-24): "))
           y = int(input(f"Стена {i+1} Y (0-24): "))
           perm = float(input("Проницаемость (0.25-1.0): "))
           grid[x][y] = 1
           walls.append(Wall(x, y, perm))
       
       # Ввод источников
       if task_type in [1, 2]:
           num_sources = int(input("Количество источников (1-5): "))
       else:
           num_sources = 1
           
       for i in range(num_sources):
           x = int(input(f"Источник {i+1} X (0-24): "))
           y = int(input(f"Источник {i+1} Y (0-24): "))
           volume = float(input("Громкость (0.1-1.0): "))
           frequency = float(input("Частота (0.1-1.0): "))
           sources.append(SoundSource(x, y, volume, frequency))
       
       # Позиция агента
       agent_x = int(input("Агент X (0-24): "))
       agent_y = int(input("Агент Y (0-24): "))
       
       return Environment(grid, (agent_x, agent_y), sources, walls, task_type)
   ```

3. **Реализовать RL-алгоритм (DQN)**
   - Архитектура нейросети:
     ```python
     class DQNAgent(nn.Module):
         def __init__(self, input_size, output_size):
             super().__init__()
             # input_size: размер MFCC + дополнительные признаки
             self.fc1 = nn.Linear(input_size, 128)
             self.fc2 = nn.Linear(128, 64)
             self.fc3 = nn.Linear(64, output_size)  # 5 действий
         
         def forward(self, x):
             x = torch.relu(self.fc1(x))
             x = torch.relu(self.fc2(x))
             return self.fc3(x)
     ```

## **ЭТАП 6: Визуализация и интерфейс (2 дня)**

### Компоненты:
- Matplotlib (базовая визуализация)
- Pygame (интерактивный интерфейс, опционально)

### Задачи:
1. **Создать функцию визуализации**
   ```python
   def visualize_environment(env, sound_map=None):
       plt.figure(figsize=(10, 10))
       grid_vis = env.grid.copy()
       
       # Отметить агента
       grid_vis[env.agent_pos[0]][env.agent_pos[1]] = 2
       
       # Отметить источники
       for source in env.sources:
           grid_vis[source.x][source.y] = 3
       
       # Цветовая карта для звука, если есть
       if sound_map is not None:
           plt.imshow(sound_map, cmap='hot', alpha=0.7)
       
       plt.imshow(grid_vis, cmap='viridis', alpha=0.3)
       plt.colorbar()
       plt.title(f"Task: {env.task_type}, Step: {env.current_step}")
       plt.savefig(f"step_{env.current_step}.png")
       plt.close()
   ```

2. **Создать консольный интерфейс для тестирования**
   ```python
   def test_interface():
       print("Выберите задачу:")
       print("1 - Поиск всех источников")
       print("2 - Найти самое тихое место")
       print("3 - Двигаться за источником")
       
       task = int(input("Ваш выбор: "))
       mode = input("Режим: (r)андомный или (m)ручной: ")
       
       if mode.lower() == 'm':
           env = manual_environment_setup(task)
       else:
           env = generate_random_environment(task)
       
       # Загрузка обученной модели
       agent = load_trained_agent(task)
       
       # Интерактивное тестирование
       while not env.done:
           action = agent.get_action(env.get_observation())
           env.step(action)
           visualize_environment(env, env.sound_map)
           print(f"Шаг: {env.current_step}, Ревард: {env.total_reward}")
   ```

## **ЭТАП 7: Обучение и оптимизация (3-5 дней)**

### Компоненты:
- Все предыдущие компоненты
- Система логирования

### Задачи:
1. **Обучение для каждой задачи отдельно**
   ```python
   def train_task(task_type, num_episodes=1000):
       agent = DQNAgent(input_size=observation_size, output_size=5)
       optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
       
       for episode in range(num_episodes):
           env = generate_random_environment(task_type)
           total_reward = 0
           
           while not env.done:
               obs = env.get_observation()
               action = agent.select_action(obs, epsilon)  # epsilon-greedy
               next_obs, reward, done = env.step(action)
               
               # Сохранить в буфер воспроизведения
               replay_buffer.add(obs, action, reward, next_obs, done)
               
               # Обучение из буфера
               if len(replay_buffer) > batch_size:
                   batch = replay_buffer.sample(batch_size)
                   train_step(agent, batch, optimizer)
               
               total_reward += reward
           
           print(f"Episode {episode}, Total Reward: {total_reward}")
           if episode % 100 == 0:
               save_agent(agent, task_type, episode)
   ```

2. **Гиперпараметры для каждой задачи**
   - Задача 1: epsilon_decay = 0.99, learning_rate = 0.001
   - Задача 2: epsilon_decay = 0.95, learning_rate = 0.0005  
   - Задача 3: epsilon_decay = 0.98, learning_rate = 0.0008

3. **Сбор статистики и анализ**
   - Графики обучения для каждой задачи
   - Тестирование на фиксированных средах для сравнения

## **ЭТАП 8: Финальная интеграция и тестирование (1-2 дня)**

### Задачи:
1. **Интегрировать все компоненты в единую систему**
2. **Создать скрипт запуска:**
   ```python
   if __name__ == "__main__":
       parser = argparse.ArgumentParser()
       parser.add_argument('--mode', choices=['train', 'test'], required=True)
       parser.add_argument('--task', type=int, choices=[1, 2, 3], required=True)
       parser.add_argument('--setup', choices=['random', 'manual'], default='random')
       
       args = parser.parse_args()
       
       if args.mode == 'train':
           train_task(args.task)
       else:  # test
           if args.setup == 'manual':
               test_interface_manual(args.task)
           else:
               test_interface_random(args.task)
   ```

3. **Провести финальное тестирование всех сценариев**

## **Общий график реализации:**
- **Неделя 1**: ЭТАПЫ 1-2 (базовая среда + физика звука)
- **Неделя 2**: ЭТАПЫ 3-4 (наблюдения + три задачи)  
- **Неделя 3**: ЭТАПЫ 5-6 (обучение + визуализация)
- **Неделя 4**: ЭТАПЫ 7-8 (оптимизация + финальная интеграция)

## **Ключевые файлы структуры:**
```
project/
├── core/
│   ├── grid_world.py       # Базовая среда
│   ├── sound_physics.py    # Физика звука
│   ├── agent.py           # Агент и его наблюдения
│   └── tasks.py           # Три задачи
├── rl/
│   ├── dqn.py             # DQN реализация
│   ├── replay_buffer.py   # Буфер воспроизведения
│   └── training.py        # Процесс обучения
├── utils/
│   ├── visualization.py   # Визуализация
│   ├── audio_processing.py # Обработка аудио
│   └── environment_gen.py # Генерация сред
├── interface/
│   ├── console_ui.py      # Консольный интерфейс
│   └── manual_setup.py    # Ручная настройка
├── models/                # Сохраненные модели
└── main.py               # Точка входа
```