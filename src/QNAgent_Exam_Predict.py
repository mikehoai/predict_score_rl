import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # Add this import

# Load the dataset
input_data = pd.read_csv('diemthi2024.csv')
# Chỉ lấy 5000 bản ghi đầu tiên
train_size = 5000
data = input_data[:train_size]
# Preprocess the data
data = data.fillna(0)  # Fill missing values with 0
X = data[['Toan', 'NguVan', 'NgoaiNgu']]
y = data.drop(columns=['sbd', 'vat_li', 'hoa_hoc', 'sinh_hoc', 'lich_su', 'dia_li', 'gdcd', 'ma_ngoai_ngu'])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the DQN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1]))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Train the model using DQN with multiple episodes and hyperparameter tuning
class DQNAgent:
    def __init__(self, model):
        self.model = model
        self.history = None  # Add this line to store training history

    def train(self, X_train, y_train, epochs, batch_size):
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)

# Hyperparameters
episodes = 10
batch_sizes = [16, 32, 64]
best_loss = float('inf')
best_params = {}
all_rewards = []  # Add this line to store rewards
all_losses = []  # Add this line to store losses

for episode in range(episodes):
    for batch_size in batch_sizes:
        agent = DQNAgent(model)
        agent.train(X_train, y_train, epochs=50, batch_size=batch_size)
        
        # Evaluate the model
        loss = agent.evaluate(X_test, y_test)
        #print(f'Episode: {episode+1}, Batch Size: {batch_size}, Test Loss: {loss}')
        
        # Save the best model
        if loss < best_loss:
            best_loss = loss
            best_params = {'episode': episode+1, 'batch_size': batch_size}
        
        # Store rewards and losses
        all_rewards.append(agent.history.history['val_loss'])
        all_losses.append(loss)

#print(f'Best Parameters: {best_params}, Best Loss: {best_loss}')

# Plot rewards and losses
plt.figure(figsize=(12, 6))

# Plot rewards
plt.subplot(1, 2, 1)
for i, rewards in enumerate(all_rewards):
    plt.plot(rewards)
plt.title('Rewards over Episodes')
plt.xlabel('Epochs')
plt.ylabel('Reward')
plt.legend()

# Plot losses
plt.subplot(1, 2, 2)
plt.plot(all_losses, label='Loss')
plt.title('Loss over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('reward-loss.png')
plt.show()

# Predict scores based on average scores of the previous year using the best model
average_scores = np.array([[6.45, 7.23, 5.51]]) # Điểm trung bình 3 môn Toan, Ngữ Văn, Ngoại Ngữ năm 2024
average_scores_df = pd.DataFrame(average_scores, columns=['Toan', 'NguVan', 'NgoaiNgu'])  # Convert to DataFrame with feature names
average_scores_scaled = scaler.transform(average_scores_df)
predicted_scores = agent.predict(average_scores_scaled)
print(f'Điểm dự đoán năm nay: {predicted_scores}')