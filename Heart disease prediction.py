{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0cd3573c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "28902e6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(r'C:\\Users\\Aditi k\\OneDrive\\Desktop\\aditi\\courses\\ML HCF notes\\15 projects HCF\\11.Heart disease prediction\\data-11.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e4083f8f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 918 entries, 0 to 917\n",
      "Data columns (total 12 columns):\n",
      " #   Column          Non-Null Count  Dtype  \n",
      "---  ------          --------------  -----  \n",
      " 0   Age             918 non-null    int64  \n",
      " 1   Sex             918 non-null    object \n",
      " 2   ChestPainType   918 non-null    object \n",
      " 3   RestingBP       918 non-null    int64  \n",
      " 4   Cholesterol     918 non-null    int64  \n",
      " 5   FastingBS       918 non-null    int64  \n",
      " 6   RestingECG      918 non-null    object \n",
      " 7   MaxHR           918 non-null    int64  \n",
      " 8   ExerciseAngina  918 non-null    object \n",
      " 9   Oldpeak         918 non-null    float64\n",
      " 10  ST_Slope        918 non-null    object \n",
      " 11  HeartDisease    918 non-null    int64  \n",
      "dtypes: float64(1), int64(6), object(5)\n",
      "memory usage: 86.2+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "2d98e65e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Sex               2\n",
       "ChestPainType     4\n",
       "RestingECG        3\n",
       "ExerciseAngina    2\n",
       "ST_Slope          3\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.select_dtypes(include=[object]).apply(lambda x: x.nunique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d3afacf2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "le = LabelEncoder()\n",
    "for col in df.select_dtypes(include='object').columns:\n",
    "    df[col] = le.fit_transform(df[col])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "9e9ebb01",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 918 entries, 0 to 917\n",
      "Data columns (total 12 columns):\n",
      " #   Column          Non-Null Count  Dtype  \n",
      "---  ------          --------------  -----  \n",
      " 0   Age             918 non-null    int64  \n",
      " 1   Sex             918 non-null    int64  \n",
      " 2   ChestPainType   918 non-null    int64  \n",
      " 3   RestingBP       918 non-null    int64  \n",
      " 4   Cholesterol     918 non-null    int64  \n",
      " 5   FastingBS       918 non-null    int64  \n",
      " 6   RestingECG      918 non-null    int64  \n",
      " 7   MaxHR           918 non-null    int64  \n",
      " 8   ExerciseAngina  918 non-null    int64  \n",
      " 9   Oldpeak         918 non-null    float64\n",
      " 10  ST_Slope        918 non-null    int64  \n",
      " 11  HeartDisease    918 non-null    int64  \n",
      "dtypes: float64(1), int64(11)\n",
      "memory usage: 86.2 KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "0a8fd561",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>918.0</td>\n",
       "      <td>53.510893</td>\n",
       "      <td>9.432617</td>\n",
       "      <td>28.0</td>\n",
       "      <td>47.00</td>\n",
       "      <td>54.0</td>\n",
       "      <td>60.0</td>\n",
       "      <td>77.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <td>918.0</td>\n",
       "      <td>0.789760</td>\n",
       "      <td>0.407701</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.00</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ChestPainType</th>\n",
       "      <td>918.0</td>\n",
       "      <td>0.781046</td>\n",
       "      <td>0.956519</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>RestingBP</th>\n",
       "      <td>918.0</td>\n",
       "      <td>132.396514</td>\n",
       "      <td>18.514154</td>\n",
       "      <td>0.0</td>\n",
       "      <td>120.00</td>\n",
       "      <td>130.0</td>\n",
       "      <td>140.0</td>\n",
       "      <td>200.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Cholesterol</th>\n",
       "      <td>918.0</td>\n",
       "      <td>198.799564</td>\n",
       "      <td>109.384145</td>\n",
       "      <td>0.0</td>\n",
       "      <td>173.25</td>\n",
       "      <td>223.0</td>\n",
       "      <td>267.0</td>\n",
       "      <td>603.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>FastingBS</th>\n",
       "      <td>918.0</td>\n",
       "      <td>0.233115</td>\n",
       "      <td>0.423046</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>RestingECG</th>\n",
       "      <td>918.0</td>\n",
       "      <td>0.989107</td>\n",
       "      <td>0.631671</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.00</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>MaxHR</th>\n",
       "      <td>918.0</td>\n",
       "      <td>136.809368</td>\n",
       "      <td>25.460334</td>\n",
       "      <td>60.0</td>\n",
       "      <td>120.00</td>\n",
       "      <td>138.0</td>\n",
       "      <td>156.0</td>\n",
       "      <td>202.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ExerciseAngina</th>\n",
       "      <td>918.0</td>\n",
       "      <td>0.404139</td>\n",
       "      <td>0.490992</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Oldpeak</th>\n",
       "      <td>918.0</td>\n",
       "      <td>0.887364</td>\n",
       "      <td>1.066570</td>\n",
       "      <td>-2.6</td>\n",
       "      <td>0.00</td>\n",
       "      <td>0.6</td>\n",
       "      <td>1.5</td>\n",
       "      <td>6.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ST_Slope</th>\n",
       "      <td>918.0</td>\n",
       "      <td>1.361656</td>\n",
       "      <td>0.607056</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.00</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>HeartDisease</th>\n",
       "      <td>918.0</td>\n",
       "      <td>0.553377</td>\n",
       "      <td>0.497414</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                count        mean         std   min     25%    50%    75%  \\\n",
       "Age             918.0   53.510893    9.432617  28.0   47.00   54.0   60.0   \n",
       "Sex             918.0    0.789760    0.407701   0.0    1.00    1.0    1.0   \n",
       "ChestPainType   918.0    0.781046    0.956519   0.0    0.00    0.0    2.0   \n",
       "RestingBP       918.0  132.396514   18.514154   0.0  120.00  130.0  140.0   \n",
       "Cholesterol     918.0  198.799564  109.384145   0.0  173.25  223.0  267.0   \n",
       "FastingBS       918.0    0.233115    0.423046   0.0    0.00    0.0    0.0   \n",
       "RestingECG      918.0    0.989107    0.631671   0.0    1.00    1.0    1.0   \n",
       "MaxHR           918.0  136.809368   25.460334  60.0  120.00  138.0  156.0   \n",
       "ExerciseAngina  918.0    0.404139    0.490992   0.0    0.00    0.0    1.0   \n",
       "Oldpeak         918.0    0.887364    1.066570  -2.6    0.00    0.6    1.5   \n",
       "ST_Slope        918.0    1.361656    0.607056   0.0    1.00    1.0    2.0   \n",
       "HeartDisease    918.0    0.553377    0.497414   0.0    0.00    1.0    1.0   \n",
       "\n",
       "                  max  \n",
       "Age              77.0  \n",
       "Sex               1.0  \n",
       "ChestPainType     3.0  \n",
       "RestingBP       200.0  \n",
       "Cholesterol     603.0  \n",
       "FastingBS         1.0  \n",
       "RestingECG        2.0  \n",
       "MaxHR           202.0  \n",
       "ExerciseAngina    1.0  \n",
       "Oldpeak           6.2  \n",
       "ST_Slope          2.0  \n",
       "HeartDisease      1.0  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe().T"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "52791efb",
   "metadata": {},
   "source": [
    "### SVM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "f2f372a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "01853b7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_encoded = df.copy()\n",
    "for col in ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']:\n",
    "    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])\n",
    "\n",
    "X = df_encoded.drop(\"HeartDisease\", axis=1)\n",
    "y = df_encoded[\"HeartDisease\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "ab418b22",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "b6f50ea9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVM Accuracy: 0.8641304347826086\n",
      "\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.82      0.86      0.84        77\n",
      "           1       0.89      0.87      0.88       107\n",
      "\n",
      "    accuracy                           0.86       184\n",
      "   macro avg       0.86      0.86      0.86       184\n",
      "weighted avg       0.87      0.86      0.86       184\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "clf = SVC(kernel=\"rbf\", random_state=42)\n",
    "clf.fit(X_train, y_train)\n",
    "y_pred = clf.predict(X_test)\n",
    "print(\"SVM Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred))\n",
    "\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "disp = ConfusionMatrixDisplay(confusion_matrix=cm)\n",
    "disp.plot(cmap=\"Blues\")\n",
    "plt.title(\"SVM Confusion Matrix\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d78413c7",
   "metadata": {},
   "source": [
    "### NB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "635922cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.naive_bayes import GaussianNB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "5e31451a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Naive Bayes Accuracy: 0.842391304347826\n",
      "\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.79      0.84      0.82        77\n",
      "           1       0.88      0.84      0.86       107\n",
      "\n",
      "    accuracy                           0.84       184\n",
      "   macro avg       0.84      0.84      0.84       184\n",
      "weighted avg       0.84      0.84      0.84       184\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAfIAAAHHCAYAAABEJtrOAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjUsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvWftoOwAAAAlwSFlzAAAPYQAAD2EBqD+naQAAQFdJREFUeJzt3QucjPX+wPHvMy7ruu52yYqQ+62tXFKkRU6J6K4SqlPJtYhTUVEqFZFLF9FNFxWVTjpSkWtC/TmFQlmXpWKtS2tl5//6/mrmzOwuZnZ2duaZ5/M+r+fYeeby/GZ22u/z/f4uj+V2u90CAABsyRXpBgAAgPwjkAMAYGMEcgAAbIxADgCAjRHIAQCwMQI5AAA2RiAHAMDGCOQAANgYgRwAABsjkCOXDh06mA2x5fDhw3LrrbdKYmKiWJYlQ4YMKfBj1KpVS2655ZYCf127euihh8xnDYQTgdymZs+ebf5AlChRQnbt2pXrfg3ETZo0ETvRNut78mzFixeX2rVry+233y6pqakSCzIyMuThhx+W5s2bS5kyZaRkyZLm93TffffJ7t27w3rsxx57zHxv7rzzTnnttdfkpptuklj770G3ZcuW5bpfV6JOSkoy919++eX5/vzmz59fAK0FClbRAn49FLJjx47J448/LlOmTCmw1/zPf/4jkVKjRg0ZP368+TkrK0u+//57mTFjhnz66afyww8/SKlSpcSutm3bJikpKbJjxw65+uqrzQmKnqz83//9n8ycOVPmzZsnW7ZsCdvxP//8c2ndurWMGTMmbMfYvHmzuFyRyw/0xHbOnDnSrl07v/1LliyRnTt3SlxcXL5fWwP5VVddJT169Aj4OQ888ICMHDky38cEAkEgt7kWLVrIiy++KKNGjZLq1asXyGtqcImUcuXKyY033ui3T7Pyu+++W5YvXy6dOnUSO/rzzz+lZ8+esnfvXvnyyy9zBZpHH31UnnjiibC2Yd++fdKoUaOwHiOUQFkQ/vGPf8jcuXNl8uTJUrTo//68aXBPTk6W3377rVDaceTIESldurRpg287gHCgtG5z//rXv+TEiRMmKz+dWbNmSceOHaVq1armD67+UZ8+ffop+8g18OgfIi0H55V9aanyueee8+5LT083fa9axtRj1K1b1wSo7OzsfL9H7dNVvn8Qf/nlF7nrrrukfv36pjxdqVIlk+X+/PPPfhmwtm/ixIm5XnPFihXmvjfffNO7T7so+vXrJwkJCabtjRs3lpdffjnXc7X6ofdpdaBChQpy7rnnmkBxKu+995589913cv/99+cK4io+Pt4Ec18akDT46PurXLmyOcHJ2Y2i/dFaotf9minqz1WqVJF7773XfC+Unjjoe92+fbt8/PHH3hK0flaekrTv5+b7HP3X48cff5RevXqZ34dmvlo9ue666+TgwYOn7CPX34P+bipWrGg+M60KaDvyOt4777xjPgd9bT3GJZdcIj/99JME6vrrr5fff/9dFi1a5N2nlZ13331Xbrjhhjyf89RTT0nbtm3Nd0g/a/3M9fG+tG0anF955RXv5+d5n55+cK0e6TH0O+H5HefsI9f/BvV2zu+VZvu6/9///nfA7xXwIJDbnGarN998s8nKT9fHqkH7zDPPNMH/6aefNsFWg+HUqVNP+hwNau3btzd/YHN6++23pUiRIuaPtDp69Kh57Ouvv27apFnRBRdcYKoFw4YNC+j9aPDRrEm3PXv2mHKwloL1hEBfy2PNmjUmGGsg0ePccccdsnjxYnMCou1QZ511lnnOG2+8kes4uq9s2bLSvXt37wmLBpjPPvvMZP/PPvusOWb//v1l0qRJ3ufp5zxo0CBzEqT79QRHqyKrV68+5fv68MMPzb+B9ktrgL3mmmvM56tdDbfddpu8//77JkDoyVLOz6xLly4mEGlQ0t+B/n5feOEFc3/Dhg1Nn7ieDGhb9WfdNOAHSoOhHmPVqlUycOBA853RrgEN0jnb40s/Vw2S2jWi3zUN0pmZmXLFFVeYroSc9IRU9+uJiH5v9Hi9e/cOuJ16ItGmTRu/E7RPPvnEnGzodyUv+rtu2bKlPPLIIyag6gmjfqd9Tzb089KTuwsvvND7+f3zn//0ex19jn739DX095WXvn37mj56/e/BM+5jw4YN5nuk3zWtKABB0+uRw35mzZql15F3r1mzxr1161Z30aJF3YMGDfLe3759e3fjxo39nnP06NFcr9OlSxf3WWed5bdPn6ubx/PPP2+OtWHDBr/HNWrUyN2xY0fv7bFjx7pLly7t3rJli9/jRo4c6S5SpIh7x44dp3xPekw9Ts6tYcOG7m3btp32vaxcudI8/tVXX83V9h9++MG7Lysry125cmV3nz59vPv69+/vrlatmvu3337ze83rrrvO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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "clf = GaussianNB()\n",
    "clf.fit(X_train, y_train)\n",
    "y_pred = clf.predict(X_test)\n",
    "print(\"Naive Bayes Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred))\n",
    "\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "disp = ConfusionMatrixDisplay(confusion_matrix=cm)\n",
    "disp.plot(cmap=\"Greens\")\n",
    "plt.title(\"Naive Bayes Confusion Matrix\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d19a18d",
   "metadata": {},
   "source": [
    "### DT"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "b508dabc",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "862756f0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Decision Tree Accuracy: 0.7880434782608695\n",
      "\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.71      0.83      0.77        77\n",
      "           1       0.86      0.76      0.81       107\n",
      "\n",
      "    accuracy                           0.79       184\n",
      "   macro avg       0.79      0.79      0.79       184\n",
      "weighted avg       0.80      0.79      0.79       184\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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n35a0ps2lZr16pl1GrjVypdeki4DBkjq6afLv95/X1648065efp0SaldO8KtBo5t9dzHJb5yogxe+JEUHTkirnLlZMX0B+TLdxZGummwSUaeWyL2xMfHm+V4tItSM2+tRGuJfcOGDVJYWCidO3f2btOkSROpW7eurF27NqBAbqsOzUmTJpnyg2dJT0+PdJNi3tuzZsnu7dul16hR3nVaIlJn/fnP0rpzZ6nVoIFk3nijpNSpI58vXx7B1gIn1rzL5dLikp7y+v/dKk/27iqL7h4q5153s7T6y/+6iRDjp2i1gli0eyY93ScWaWw6li+//NL0f2ugv/nmm2XRokXSrFkzyc7Olri4OKlatarP9qmpqea+QNgqIx87dqyMGDHCe1uPigjm4fPOrFny7WefSd9JkySpRg3v+sRq1cy/p5R477WPfP/PP5d5O4FAdBl2l6x+9gnZ/M83zO3dW7dI1bQ68qf+t8kXSxdEunmwgR07dpjBax7Hy8YbN24sn3/+uezfv19ee+016du3r6xcuTKk7bFVIPenfIHgaca97MknZcvatXL9pElSLS3N5/6qqalSpXp1+fWnn3zW79m5Uxq0bVvGrQUCUyGhoukyKq6o6IhYUTq1CKFmBf0MnlHo/tCsu2HDhubvtm3bymeffSaPPvqoXH311VJQUCD79u3zycp11Hpaid/cmCqto2y8M3Om/OvDD+WKkSMlvmJFObh3r1kK/xifoFMFM3r2lE+XLJGvPv7YBPAPXnxRfvnxR2nTpUukmw8c1zerlssFA4fKGedfbEalN7noEsno83f5+oP/TQuqmFRV0ho1l1PqNzK3U05rYG7rvHPYmFW2089KU1RUZMZ6aVCvUKGCrFjx++wJlZWVJdu3bzd96LbJyA8ePChbt2713t62bZspQVSvXt10+CMy1v8xz/H5//s/n/WXDx1q+sRVh+7d5beCAjMN7b8HDpgBb33uvVeq16oVkTYD/nr7oX9Ip1tHy2VjH5DK1VLMCWE2vP6CrHxqinebxh27So97HvXevvKBJ82/Hz75sHz45CMRaTfsZ+zYsWZmlsazAwcOmBHqH374obz77rumb33gwIGmu1hjnmb4Q4YMMUE8kIFuynJ7Ri5FgL6giy666Kj12ofw7LPPnvDx2keub8beja9JUmLlMLUSiKx7S0xhAWJJ/hG3PLD1iOlD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umlA0A4ePChbt2713t62bZt8/vnnUr16dalbt25E2wacCNPP4DedejZ58mQzwK1169by2GOPmWlpgN19+OGHctFFFx21Xg9en3322Yi0CfAXgRwAABujjxwAABsjkAMAYGMEcgAAbIxADgCAjRHIAQCwMQI5AAA2RiAHAMDGCORAkPr16+dz7eoLL7xQhg0bFpGTmui1tPft23fMbfT+xYsX+/2c48ePNyf/CcZ//vMfs189UxqA0COQI2aDqwYPXfRc8Q0bNpR7771Xfvvtt7Dve+HChXLfffeFLPgCwPFwrnXErD//+c8yd+5cyc/Pl7ffflsGDRokFSpUkLFjxx61bUFBgQn4oaDn5waAskJGjpgVHx9vLrNar149ueWWW6Rz587y5ptv+pTDJ0yYILVr15bGjRub9Tt27JCrrrpKqlatagJy9+7dTWnY48iRI+YCMnp/SkqKjBo1Skqe5bhkaV0PJEaPHi3p6emmTVodmD17tnlez/m9q1WrZjJzbZfn6nKTJk2S008/XSpWrCitWrWS1157zWc/enDSqFEjc78+T/F2+kvbpc9RqVIlqV+/vtx1111SWFh41HZPPvmkab9up+/P/v37fe5/5plnpGnTppKQkCBNmjSRGTNmBNwWACeHQA7H0ICnmbeHXoY1KytLli9fLkuXLjUBLDMzU6pUqSIfffSRfPzxx5KYmGgye8/jHnnkEXMRjTlz5sjq1atlz549smjRouPu9/rrr5eXXnrJXGTm66+/NkFRn1cD4+uvv2620Xbs2rVLHn30UXNbg/jzzz8vs2bNkn//+98yfPhw6dOnj6xcudJ7wNGzZ0/p1q2b6Xu+4YYbZMyYMQG/J/pa9fV89dVXZt9PP/20TJ061WcbvSrYq6++KkuWLJFly5bJpk2b5NZbb/XeP2/ePBk3bpw5KNLXN3HiRHNA8NxzzwXcHgAnQS+aAsSavn37urt3727+Lioqci9fvtwdHx/vHjlypPf+1NRUd35+vvcxL7zwgrtx48Zmew+9v2LFiu53333X3K5Vq5b7oYce8t5fWFjorlOnjndfqmPHju6hQ4eav7OysjRdN/svzQcffGDu37t3r3ddXl6eu1KlSu41a9b4bDtw4ED3tddea/4eO3asu1mzZj73jx49+qjnKknvX7Ro0THvnzx5srtt27be23fffbe7XLly7h9//NG77p133nG7XC73rl27zO0GDRq458+f7/M89913nzsjI8P8vW3bNrPfTZs2HXO/AE4efeSIWZpla+armbaWqv/2t7+ZUdgeLVq08OkX/+KLL0z2qVlqcXl5efLdd9+ZcrJmzcUv3Vq+fHlzjfZjXURQs+Vy5cpJx44d/W63tuHw4cPSpUsXn/VaFWjTpo35WzPfkpeQzcjIkEC98sorplKgr0+vya2DAZOSkny20etxn3rqqT770fdTqwj6XuljBw4cKDfeeKN3G32e5OTkgNsDIHAEcsQs7TeeOXOmCdbaD65Bt7jKlSv73NZA1rZtW1MqLumUU0456XJ+oLQd6q233vIJoEr72ENl7dq10rt3b7nnnntMl4IG3pdfftl0HwTaVi3Jlzyw0AMYAOFHIEfM0kCtA8v8ddZZZ5kMtWbNmkdlpR61atWSTz75RC644AJv5rlhwwbz2NJo1q/Zq/Zt62C7kjwVAR1E59GsWTMTsLdv337MTF4HlnkG7nmsW7dOArFmzRozEPAf//iHd90PP/xw1Hbajp07d5qDIc9+XC6XGSCYmppq1n///ffmoABA2WOwG/AHDUQ1atQwI9V1sNu2bdvMPO/bbrtNfvzxR7PN0KFD5YEHHjAnVdmyZYsZ9HW8OeCnnXaa9O3bVwYMGGAe43lOHTymNJDqaHXtBvj5559Nhqvl6pEjR5oBbjpgTEvXGzdulMcff9w7gOzmm2+Wb7/9Vu644w5T4p4/f74ZtBaIM844wwRpzcJ1H1piL23gno5E19egXQ/6vuj7oSPXdUaA0oxeB+fp47/55hv58ssvzbS/KVOmBNQeACeHQA78QadWrVq1yvQJ64hwzXq171f7yD0Z+u233y7XXXedCWzaV6xB94orrjju82p5/69//asJ+jo1S/uSDx06ZO7T0rkGQh1xrtnt4MGDzXo9oYyO/NYAqe3QkfNaatfpaErbqCPe9eBAp6bp6HYdLR6Iyy+/3Bws6D717G2aoes+S9Kqhr4fl156qXTt2lVatmzpM71MR8zr9DMN3lqB0CqCHlR42gogvCwd8RbmfQAAgDAhIwcAwMYI5AAA2BiBHAAAGyOQAwBgYwRyAABsjEAOAICNEcgBALAxAjkAADZGIAcAwMYI5AAA2BiBHAAAGyOQAwAg9vX/abD53v8O2KAAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "clf = DecisionTreeClassifier(random_state=42)\n",
    "clf.fit(X_train, y_train)\n",
    "y_pred = clf.predict(X_test)\n",
    "\n",
    "print(\"Decision Tree Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred))\n",
    "\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "disp = ConfusionMatrixDisplay(confusion_matrix=cm)\n",
    "disp.plot(cmap=\"Oranges\")\n",
    "plt.title(\"Decision Tree Confusion Matrix\")\n",
    "plt.show()\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8eccf907",
   "metadata": {},
   "source": [
    "### RF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "555b2ca6",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "970c75f9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Accuracy: 0.8804347826086957\n",
      "\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.86      0.86      0.86        77\n",
      "           1       0.90      0.90      0.90       107\n",
      "\n",
      "    accuracy                           0.88       184\n",
      "   macro avg       0.88      0.88      0.88       184\n",
      "weighted avg       0.88      0.88      0.88       184\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAfIAAAHHCAYAAABEJtrOAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjUsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvWftoOwAAAAlwSFlzAAAPYQAAD2EBqD+naQAAP05JREFUeJzt3QmcjfX+wPHvM5YZYcYSM2QsIcu1lYopEQ2TutKlzdVNSMtFoRT/ohJpuyll6ZZoE6kounGlsu/U1WJQYsSgMGObGWbO//X96ZzOGYNz5sxynvN83r2eZs6z/s4zx/k+39/yPJbL5XIJAACwpYjiLgAAAMg/AjkAADZGIAcAwMYI5AAA2BiBHAAAGyOQAwBgYwRyAABsjEAOAICNEcgBALAxAjnO6c4775TatWsXdzFQhI4cOSJ33XWXxMXFiWVZMmjQoAI/hn6m9LOFU5544glzroFAEchDyLRp08w/ZPdUsmRJueCCC8yX3a+//lrcxQvZ8+Q9DRs2TELR008/LXPmzAlom/T0dHnyySelefPmUq5cOSlTpow0adJEHnnkEdm9e7cUdnn1PN93333yzjvvyD/+8Q8Jx8/PsmXLTluud62Oj483y//6178W2d8byK+S+d4ShWbUqFFSp04dycjIkFWrVpkvHv3C+e677yQqKqq4ixdy58mbBrpQpF/sN910k9x4441+rf/zzz9LYmKi7Ny5U26++Wa5++67pXTp0vK///1PpkyZIrNnz5YtW7YUWnm//PJLad26tTz++OOFdozk5GSJiCi+XEL/LU2fPl3atGnjM3/x4sWya9cuiYyMLLK/t3rsscdC9kIUoY1AHoI6d+4sl156qfldqzfPP/98efbZZ+XTTz+VW265pbiLF5LnqSAdPXpUypYtK8Xl5MmT0q1bN9m7d698/fXXpwWaMWPGmM9DYdq3b580bty4UI8RTKAsCNddd53MmjVLxo8fb2q/3DS4t2zZUn777bciKYf786Zl8C4H4C+q1m3gqquuMj9/+uknz7ysrCwZOXKk+cKJiYkxXwS63ldffeWz7S+//GKqCF944QX597//LXXr1jVfoJdddpmsXbv2tGNpdaBmtZqt6E/N/M705fPggw+aKkjdX4MGDcwxcj9MT489YMAA84WpgUGrhxMSEmTTpk1m+WuvvSb16tUzx7v66qtNeQsyq9RzouemQoUK0rVrV/nxxx/zbJf84Ycf5O9//7tUrFjRJ3C+++675hxruStVqiS33XabpKSk+Oxj69at0r17d9OerO+jRo0aZr20tDTPOdDz9dZbb3mqdM/WNvzRRx/Jt99+K48++uhpQVxFR0ebYO5Nz6+7nHrhd/vtt5/WHKPH1Cp6na+Zov5epUoVeeihhyQ7O9usoxcOWr7t27fLZ5995imv/l3cVdK5/0bubfSnv+fkTG3kWhOhNRB6rs877zxTK6DlyOt4H3zwgTkPum89xjXXXCPbtm0Tf/Xo0UN+//13Wbhwoc+/qw8//NB8FvKin/ErrrhCKleubM61nnNd39vZ/t5n+7zlbiOfOnWqef3mm2+elu3r/P/85z9+v1eENy7/bMD9xan/6L3bT9944w3zZdSvXz85fPiwqXJNSkqSNWvWSIsWLXz2oVmGrnPPPfeYL4HnnnvOZH36xVmqVCmzzn//+1/z5asBd+zYseZLrnfv3uaL0psG6xtuuMFcNPTt29cca8GCBTJ06FATJMaNG+ez/tKlS01tQv/+/c1r3be2PT788MMyceJE+ec//ykHDx40ZerTp48JwP7QoJA7a9Igpr744guTsV944YXmC/L48ePyyiuvyJVXXikbNmw4rfOeBo/69eubL0n3xYgGiREjRphaEK0Z2b9/v9lH27ZtZePGjebiQL/49ZxnZmbKwIEDTeDSczBv3jw5dOiQucjSNmbd/vLLLzdV5EovqM5Ez5Xyt11aA6z+nfTiTM+tZvIvv/yyLF++3FNONw3YWt5WrVqZoKTn6V//+pcpj7aHN2rUyJR38ODB5u+uF2tKA76//DknedFya5A8duyY3H///SZYajDUz5oGy7/97W8+6z/zzDOmal4vRPSzoJ+fnj17yurVq/0qp34G9KLy/fffN58V9fnnn5t96UWHZuq56XnV8uhx9H3OmDHDfHb0vV1//fVmHX/+3nl93nLTv+nHH38sQ4YMkY4dO5qLZr0A1n4T+u9OaxQAQ59HjtAwdepU/Rft+uKLL1z79+93paSkuD788ENXlSpVXJGRkea128mTJ12ZmZk+2x88eNAVGxvr6tOnj2fe9u3bzT4rV67sOnDggGf+J598YubPnTvXM69FixauatWquQ4dOuSZ99///tesV6tWLc+8OXPmmHmjR4/2Of5NN93ksizLtW3b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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "clf = RandomForestClassifier(random_state=42)\n",
    "clf.fit(X_train, y_train)\n",
    "y_pred = clf.predict(X_test)\n",
    "print(\"Random Forest Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred))\n",
    "\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "disp = ConfusionMatrixDisplay(confusion_matrix=cm)\n",
    "disp.plot(cmap=\"Purples\")\n",
    "plt.title(\"Random Forest Confusion Matrix\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "03ace51a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 800x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model Accuracies: {'SVM': 0.8641304347826086, 'Naive Bayes': 0.842391304347826, 'Decision Tree': 0.7880434782608695, 'Random Forest': 0.8804347826086957}\n"
     ]
    }
   ],
   "source": [
    "accuracies = {\n",
    "    \"SVM\": accuracy_score(y_test, SVC(kernel=\"rbf\", random_state=42).fit(X_train,y_train).predict(X_test)),\n",
    "    \"Naive Bayes\": accuracy_score(y_test, GaussianNB().fit(X_train,y_train).predict(X_test)),\n",
    "    \"Decision Tree\": accuracy_score(y_test, DecisionTreeClassifier(random_state=42).fit(X_train,y_train).predict(X_test)),\n",
    "    \"Random Forest\": accuracy_score(y_test, RandomForestClassifier(random_state=42).fit(X_train,y_train).predict(X_test))\n",
    "}\n",
    "\n",
    "plt.figure(figsize=(8,6))\n",
    "plt.bar(accuracies.keys(), accuracies.values(), color=[\"blue\",\"green\",\"orange\",\"purple\"])\n",
    "plt.ylabel(\"Accuracy\")\n",
    "plt.xlabel(\"Models\")\n",
    "plt.title(\"Model Accuracy Comparison\")\n",
    "plt.ylim(0,1)\n",
    "plt.show()\n",
    "\n",
    "print(\"Model Accuracies:\", accuracies)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b312717e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
