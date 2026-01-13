# Перенос проекта в Google Colab

## Общая информация
Проект реализует систему навигации по звуковым ориентирам с использованием методов обучения с подкреплением (DQN). Для работы в Google Colab необходимо учитывать особенности облачной среды.

## Структура проекта
```
/workspace/
├── core/                   # Основная логика среды и агента
│   ├── grid_world.py       # Среда сеточного мира и классы агента
│   ├── sound_source.py     # Логика звуковых источников и распространения
│   └── tasks.py            # Специфические задачи навигации
├── rl/                     # Компоненты обучения с подкреплением
│   ├── dqn.py             # Реализация DQN
│   └── training.py        # Утилиты обучения
├── utils/                  # Вспомогательные функции
│   ├── audio_processing.py # Извлечение аудио признаков
│   ├── environment_gen.py  # Генерация сред
│   └── visualization.py    # Визуализация
├── interface/              # Интерфейсы пользователя
│   └── console_ui.py      # Консольный интерфейс
├── main.py                # Главная точка входа
└── requirements.txt       # Зависимости
```

## Инструкция по переносу в Google Colab

### 1. Создание ноутбука
Создайте новый ноутбук в Google Colab.

### 2. Установка зависимостей
Вставьте следующий блок кода в ячейку и выполните:

```python
# Установка необходимых библиотек
!pip install numpy torch matplotlib tqdm

# Pygame может не работать корректно в Colab, поэтому для визуализации будем использовать matplotlib
# !pip install pygame  # Можно попробовать, но визуализация может не работать
```

### 3. Загрузка кода проекта
Есть несколько способов загрузить код в Colab:

#### Вариант 1: Загрузка через GitHub (если репозиторий доступен)
```python
# Клонирование репозитория
!git clone https://github.com/yourusername/your-repo-name.git
%cd your-repo-name
```

#### Вариант 2: Загрузка файлов напрямую
```python
# Загрузка файлов через интерфейс Colab
from google.colab import files
uploaded = files.upload()
```

#### Вариант 3: Создание файлов вручную в Colab
Вы можете скопировать и вставить каждый файл в отдельную ячейку с кодом для создания файлов:

```python
%%writefile grid_world.py
# Вставьте сюда содержимое файла /workspace/core/grid_world.py

%%writefile sound_source.py
# Вставьте сюда содержимое файла /workspace/core/sound_source.py

# И так далее для всех файлов...
```

### 4. Особенности работы в Colab

#### 4.1. Работа с визуализацией
Pygame не работает в Colab, поэтому модуль `utils/visualization.py` нужно адаптировать для использования matplotlib:

```python
# Вместо PygameVisualizer используйте визуализацию через matplotlib
import matplotlib.pyplot as plt
import numpy as np

def visualize_environment(env):
    """Визуализация среды с помощью matplotlib"""
    # Создаем копию сетки для визуализации
    vis_grid = env.grid.copy()
    
    # Отмечаем позицию агента
    if env.agent_pos is not None:
        x, y = env.agent_pos
        vis_grid[x][y] = 2
    
    # Отмечаем звуковые источники
    for source in env.sound_sources:
        vis_grid[source.x][source.y] = 3
    
    plt.figure(figsize=(8, 8))
    plt.imshow(vis_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Тип ячейки (0: Пусто, 1: Стена, 2: Агент, 3: Источник звука)')
    plt.title('Визуализация сеточного мира')
    plt.show()
```

#### 4.2. Запуск обучения
После установки всех файлов можно запустить обучение:

```python
# Добавляем путь к проекту
import sys
sys.path.append('/content/your-repo-name')

# Импортируем нужные модули
from rl.training import train_task, evaluate_agent
from utils.environment_gen import generate_random_environment

# Запускаем обучение для задачи 1
print("Обучение агента для задачи 1...")
agent, losses = train_task(
    task_type=1,
    num_episodes=100,  # Меньше эпизодов для тестирования в Colab
    model_path=None
)

# Оцениваем агента
print("Оценка обученного агента...")
evaluate_agent(agent, task_type=1, num_episodes=5)
```

### 5. Пример минимального ноутбука для запуска

```python
# Ячейка 1: Установка зависимостей
!pip install numpy torch matplotlib tqdm

# Ячейка 2: Создание файлов проекта
%%writefile core/__init__.py
# Пустой файл для создания пакета

%%writefile core/grid_world.py
# Вставьте сюда полное содержимое файла grid_world.py

# Повторите для всех остальных файлов...

# Ячейка 3: Запуск обучения
import sys
sys.path.append('/content/')
from rl.training import train_task
from utils.environment_gen import generate_random_environment

# Запуск обучения
agent, losses = train_task(task_type=1, num_episodes=50)
```

### 6. Замечания о производительности
- В Colab доступны GPU, что можно использовать для ускорения обучения DQN
- Для использования GPU измените runtime на GPU в настройках Colab
- Модель PyTorch автоматически будет использовать GPU, если он доступен

### 7. Альтернативные подходы
Если pygame не нужен, можно отключить визуализацию или использовать текстовый вывод:

```python
# Отключение визуализации в интерфейсе
# В файле interface/console_ui.py закомментируйте строки, связанные с PygameVisualizer
```

## Тестирование
После переноса рекомендуется протестировать:
1. Загрузку всех модулей
2. Создание среды
3. Обучение простого агента
4. Визуализацию результатов (если применимо)