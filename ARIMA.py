# КОД НАПИСАН НА ЯЗЫКЕ PYTHON v3.7
# Импортирование основных библиотек
# Остальные библиотеки импортируются по ходу исполнения программы
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.arima_model import ARIMA
from itertools import product
from tqdm import tqdm
import warnings
import numpy as np


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

### ПРИМЕЧАНИЕ : ДЛЯ ПОСТРОЕНИЯ ГРАФИКА ИСПОЛЬЗУЕТСЯ МЕТОД plot() 
### БИБЛИОТЕКИ matplotlib
### ЗА ДОП. ИНФОЙ СМОТРЕТЬ ДОКУМЕНТАЦИЮ

# Загрузка данных из .csv
file_path = r'D:\Downloads\BYN.csv'
BYN = pd.read_csv(file_path, sep = ',')


first_obs = 5453+ 2*31
last_obs = 5453 + 365
# Выбор данных
FRAME = BYN[first_obs : last_obs]['CLOSE']
FRAME = FRAME.loc[5453 + 2*31:]
train = FRAME[:-31]
test = FRAME[-31:]

### ЧТОБЫ РАБОТАТЬ СО СКОЛЬЗЯЩЕЙ СРЕДНЕЙ,
### ЗАМЕНИТЬ ПРЕДЫДУЩИЙ ВЫБОР ДАННЫХ НА ЭТОТ
#iMA = FRAME[-365:].rolling(window=7).mean().dropna()
#train = iMA[:-31]
#test = iMA[-31:]
#test_real = FRAME[-37:] ### ДЛЯ ВОССТАНОВЛЕНИЯ ПОСЛЕДНИХ ЗНАЧЕНИЙ РЯДА

# Вывод статистики ряда
import scipy.stats as scs
jb_test = scs.stats.jarque_bera(train)
print('JB-test: ', jb_test)
print('Variation = ', train.mean()/train.std())
print(train.describe())

# Вывод результатов теста Дики-Фуллера
from statsmodels.tsa.stattools import adfuller
result = adfuller(train.diff().dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# Дополнительные тесты на стационарность (Включая Дики-Фуллера)
from pmdarima.arima.utils import ndiffs
## Adf Test
adf_test = ndiffs(train, test='adf') 
# KPSS test
KPSS_test = ndiffs(train, test='kpss')  
# PP test:
PP_test = ndiffs(train, test='pp')
print(adf_test, KPSS_test, PP_test)

# Отрисовка рядов и Коррелограмм
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Исходный ряд
fig, axes = plt.subplots(3, 3)
axes[0, 0].plot(train.dropna()); axes[0, 0].set_title('Original Series')
plot_acf(train.dropna(), ax=axes[0, 1], lags = 25)
plot_pacf(train.dropna(), ax=axes[0, 2], lags = 50)
# d = 1
axes[1, 0].plot(train.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(train.diff().dropna(), ax=axes[1, 1])
plot_pacf(train.diff().dropna(), ax=axes[1, 2])
# d = 2
axes[2, 0].plot(train.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(train.diff().diff().dropna(), ax=axes[2, 1])
plot_pacf(train.diff().diff().dropna(), ax=axes[2, 2])

plt.show()

### ПЕРЕБОР ВСЕХ ВАРИАНТОВ (p,d,q) и ВЫБОР НАИЛУЧШЕЙ МОДЕЛИ НА ОСНОВЕ AIC-критерия ###

ps = range(0, 6) # второй параметр подбирается, основываясь на анализе Коррелограмм
qs = range(0, 6)
parameters = product(ps, qs)
parameters_list = list(parameters)
test = np.array(test)
prediction = np.array([])
# нижняя и верхняя границы дов. инт.
lower_series = np.array([])
upper_series = np.array([]) 
index = 0
results = []
best_aic = float("inf")
for param in tqdm(parameters_list):    
        # try except нужен, потому что на некоторых наборах параметров модель не обучается
    try:
    ###  ПОМЕНЯТЬ d ПРИ ИССЛЕДОВАНИИ ДРУГОГО РЯДА ###
        model=ARIMA(train, order=(param[0], 1, param[1])).fit(disp=0)        
    # выводим параметры, на которых модель не обучается и переходим к следующему набору
    except ValueError:
        print('wrong parameters:', param)
        continue
    except:
        continue            
    aic = model.aic
    results.append((model, aic, param))
    #сохраняем лучшую модель, aic, параметры
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
        warnings.filterwarnings('default')
        
### Вывод информации по модели
best_model.summary()

### Построение прогноза с постоянным дообучением
prediction = np.array([])
index = 0
#restored_values = []

while len(prediction) != len(test):
    best_model = ARIMA(train, order=(best_param[0], 1, best_param[1])).fit(disp=0)
    fc = model.forecast(1, alpha=0.05)[0]
    ### Для скользящей средней восстанавливает значения ###
    #restored_values.append(7*fc - sum(test_real.iloc[index:6+index])) 
    train = np.concatenate((train, test[index : index + 1]), axis=0)
    index += 1
    prediction = np.concatenate((prediction, fc), axis=0)

### ОТРИСОВКА ОСТАТКОВ МОДЕЛИ НА ОБУЧАЮЩЕЙ ВЫБОРКЕ ###

#residuals = pd.DataFrame(best_model.resid, columns = ['resids'])
#fig, ax = plt.subplots(1,2)
#residuals.plot(title="Residuals", ax=ax[0])
#residuals.plot(kind='kde', title='Density', ax=ax[1])
#plt.show()    

### ОТРИСОВКА ВОССТАНОВЛЕННЫХ ЗНАЧЕНИЙ ###

#plt.plot(restored_values, label='forecast')
#plt.plot(np.array(FRAME[-31:]), label='actual')
#plt.title('MAE {}'.format(mae(np.array(FRAME[-31:]), restored_values)))
#plt.legend(loc='upper left', fontsize=8)
#plt.show()

### ОТРИСОВКА ПРОГНОЗА НА ТЕСТОВОЙ ВЫБОРКЕ ###

#plt.plot(np.array(test), label='actual')
#plt.plot(prediction, label='forecast')
#plt.title('MAE {}'.format(mae(test, prediction)))
#test = pd.Series(test)
#lower_series = pd.Series(lower_series, index=test.index)
#upper_series = pd.Series(upper_series, index=test.index)
#plt.fill_between(list(range(31)), lower_series[:31], upper_series[:31], 
 #                color='k', alpha=.15)
#plt.legend(loc='upper left', fontsize=8)
#plt.show()
