import pandas as pd
import numpy as np
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from scipy import stats
from sklearn import linear_model, datasets
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\MATEI NEAGA\Desktop\Inteligenta Artificiala\Proiect\setDate.csv",
                 encoding="ISO-8859-1")


def map_gender_to_numeric(gender):
    if gender.lower() == 'male':
        return 1
    elif gender.lower() == 'female':
        return 1
    else:
        return 5


# se aplica funcția pe coloana 'Gender'
gender = df['2. Gender'].apply(map_gender_to_numeric)

status = df["3. Relationship Status"].map({
    "Single": 0,
    "Divorced": 1,
    "In a relationship": 3,
    "Married": 5
}).astype('int64')

social_media = df["7. What social media platforms do you commonly use?"]

useSocials = df['6. Do you use social media?'].map({
    'No': 1,
    'Yes': 2
}).astype('int64')


def map_social_media_count(count):
    if count <= 2:
        return 1
    elif count <= 3:
        return 2
    elif count <= 4:
        return 3
    elif count <= 5:
        return 4
    elif count <= 6:
        return 5


# Count the number of social media platforms for each row
social_media_count = social_media.apply(lambda x: len(x.split(', ')))

# Map the count to a score using the custom function
socials = social_media_count.map(
    map_social_media_count).fillna(0).astype('int64')

# etichetare, se foloseste functia map pentru a inlocui valorile de tip text cu valori numerice, pentru a se putea face corelari ulterioare
avgTime = df["8. What is the average time you spend on social media every day?"].map({
    "Less than an Hour": 0,
    "Between 1 and 2 hours": 1,
    "Between 2 and 3 hours": 2,
    "Between 3 and 4 hours": 3,
    "Between 4 and 5 hours": 4,
    "More than 5 hours": 5
}).astype('int64')

data_frames = [pd.to_numeric(
    df["1. What is your age?"].apply(lambda x: int(x))),
    pd.to_numeric(
        df["9. How often do you find yourself using Social media without a specific purpose?"]),
    pd.to_numeric(
    df["10. How often do you get distracted by Social media when you are busy doing something?"]),
    pd.to_numeric(
    df["11. Do you feel restless if you haven't used Social media in a while?"]),
    pd.to_numeric(
    df["12. On a scale of 1 to 5, how easily distracted are you?"]),
    pd.to_numeric(
    df["13. On a scale of 1 to 5, how much are you bothered by worries?"]),
    pd.to_numeric(
    df["14. Do you find it difficult to concentrate on things?"]),
    pd.to_numeric(
    df["15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?"]),
    pd.to_numeric(
    df["16. Following the previous question, how do you feel about these comparisons, generally speaking?"]),
    pd.to_numeric(
    df["17. How often do you look to seek validation from features of social media?"]),
    pd.to_numeric(
    df["18. How often do you feel depressed or down?"]),
    pd.to_numeric(
    df["19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?"]),
    pd.to_numeric(
    df["20. On a scale of 1 to 5, how often do you face issues regarding sleep?"]),

]
# "1. What is your age?""
df_1 = data_frames[0]
# "9. How often do you find yourself using Social media without a specific purpose?"
df_9 = data_frames[1]
# "10. How often do you get distracted by Social media when you are busy doing something?"
df_10 = data_frames[2]
# "11. Do you feel restless if you haven't used Social media in a while?"
df_11 = data_frames[3]
# "12. On a scale of 1 to 5, how easily distracted are you?"
df_12 = data_frames[4]
# "13. On a scale of 1 to 5, how much are you bothered by worries?"
df_13 = data_frames[5]
# "14. Do you find it difficult to concentrate on things?"
df_14 = data_frames[6]
# "15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?"
df_15 = data_frames[7]
# "16. Following the previous question, how do you feel about these comparisons, generally speaking?"
df_16 = data_frames[8]
# "17. How often do you look to seek validation from features of social media?"
df_17 = data_frames[9]
# "18. How often do you feel depressed or down?"
df_18 = data_frames[10]
# "19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?"
df_19 = data_frames[11]
# "20. On a scale of 1 to 5, how often do you face issues regarding sleep?"
iesire = data_frames[12]


# verificare uniformitate date
ok = 0
for data_frame in data_frames:
    if data_frame.dtype == 'int64':
        ok += 1
    else:
        ok -= 1


# se verifica uniformitatea datelor
if ok == len(data_frames):
    print('Datele sunt uniforme')
    # print(f'Matricea de corelatie: \n', matrice_corelatie)


print('Se exporteaza matricea de corelatie intr-un fisier tip Excel separat pentru o mai buna vizualizare a sa')
print('\n')
# se creeaza un dataframe pentru matricea de corelatie
all_features = pd.concat([df_1, gender, useSocials, socials, status, avgTime, df_9, df_10,
                         df_11, df_12, df_13, df_14, df_15, df_16, df_17, df_18, df_19], axis=1)
# matricea de corelatie a celorlalte caracteristici, in afara de iesirea Y, figurand ca lipsa somnulului (coloana 20)
matrice_corelatie_completa = np.corrcoef(all_features, rowvar=False)
# se scrie matricea de corelatie intr-un excel separat
export_matrice_corelatie_completa = pd.DataFrame(
    data=matrice_corelatie_completa)
export_matrice_corelatie_completa.to_excel(
    r'C:\Users\MATEI NEAGA\Desktop\Inteligenta Artificiala\Proiect\matrice_corelatie_completa.xlsx')


# Histograma privind calitatea somnului

plt.hist(iesire)
plt.xlabel('Probleme cu somnul de la 1 la 5')
plt.ylabel('Numar de oameni')
plt.title('Histograma privind calitatea somnului')
plt.show()

# test T intre numarul de ore petrecut pe social media si calitatea somnului

# deviatia standard
print("Deviatia standard a vectorului de valori corespunzatoare calitatii somnului este:", stats.tstd(iesire))
print("Deviatia standard a vectorului de valori corespunzatoare timpului petrecut pe social media este:", stats.tstd(avgTime))
print('\n')

# media calitatii somnului
print('Media calitatii somnului, conform notelor oferite este:', iesire.mean())
print('Media numarului de ore petrecut pe retelele sociale, conform notelor oferite este:', avgTime.mean())
print('\n')

# parametrii t si p
t, p = stats.ttest_ind(avgTime, iesire)
# intensitatea diferentei dintre grupuri
print("Valoarea t:", t)
# probabilitatea de a obtine o valoare t cel putin la fel de mare ca valoarea obtinuta anterior
print("Valoarea p:", p)
print('\n')

# numarul de valori din setul de date care pot varia pentru a putea face comparatii
print('Gradele de libertate ale primului vector de valori, corespunzator cu timpul petrecut pe social media este:', len(avgTime))
print('Gradele de libertate ale celui de-al doilea vector de valori, corespunzator cu calitatea somnului este:', len(iesire))
print('\n')

avgTime = avgTime.values.reshape(-1, 1)
iesire = iesire.values.reshape(-1, 1)
iesire = iesire.ravel()
# antrenarea si prezicerea modelului
# test_size 40% reprezinta procentul de date convertite in date de test, restul fiind date de antrenament
# random_state = 42 reprezinta intensitatea cu care sunt amestecate datele
avgTime_train, avgTime_test, iesire_train, iesire_test = train_test_split(
    avgTime, iesire, test_size=0.4, random_state=42)

# invatare supervizata de tip Logistica regresie
print('Invatare supervizata de tip regresie')

# model de regresie liniara
regr = linear_model.LinearRegression()

# antrenarea modelului
regr.fit(avgTime_train, iesire_train)

# predictia modelului
iesire_pred = regr.predict(avgTime_test)

# The coefficients
print("Coeficient de regresie: \n", regr.coef_)
# The mean squared error
print("Eroarea mediei patratice: %.2f" %
      mean_squared_error(iesire_test, iesire_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coeficientul de determinare: %.2f" %
      r2_score(iesire_test, iesire_pred))

# varianta 1

# grafic
plt.scatter(avgTime_test, iesire_test, color="black")
plt.plot(avgTime_test, iesire_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())
plt.xlabel(('avgTime'))
plt.ylabel(('calitatea somnului'))

plt.show()

# varianta2

# creare un dataframe dintre coloana de intrare si coloana de iesire
dataframe_kde = pd.DataFrame(
    {'avgTime': avgTime.ravel(), 'calitatea_somnului': iesire})

#  grafic 2D Density
print('Grafic 2D density pentru o mai buna vizualizare a iesirii in functie de intrare, intrucat o mare majoritate a notelor acordate se suprapun, drept urmare dintr-un grafic tip scatter nu se pot trage concluzii pertinente.')
sns.kdeplot(data=dataframe_kde, x='avgTime',
            y='calitatea_somnului', fill=True, cmap='Blues')

plt.plot(avgTime_test, iesire_pred, color="blue", linewidth=3)
plt.xlabel('Timpul petrecut pe social media')
plt.ylabel('Probleme privind calitatea somnului')
plt.title(
    'Grafic 2D Density între timpul petrecut pe social media și calitatea somnului')
plt.show()


# varianta 3
# plt.scatter(avgTime, iesire, alpha=0.5)  # Adjust alpha for transparency

# # Add labels and title
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Scatter Plot with Overlapping Values')

# Show the plot
# plt.show()

# varianta 4
# sns.scatterplot(x='avgTime', y='calitatea_somnului', data=dataframe_kde, alpha=0.5,
#                 color='blue', label='Dataset 1')

# # Add linear trendline
# sns.regplot(x='avgTime', y='calitatea_somnului',
#             data=dataframe_kde, scatter=False, color='blue')

# # Set labels and title
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Scatter Plot with Overlapping Values and Trendlines')

# # Show the plot
# plt.legend()
# plt.show()


# # Matrice de corelație tip HeatMap
sns.heatmap(matrice_corelatie_completa, annot=True,
            cmap='coolwarm')
plt.title('Matrice de Corelație')
plt.show()

# Algoritm Gaussian Naive Bayes

# Pregatirea datelor pentru antrenare si testare
X = all_features.values
y = iesire

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)

# Gaussian Naive Bayes
print('\n')
print('Invatare folosind algoritmul gaussian Naive Bayes')
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

# Se evalueaza performanta
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

print(f'Acuratetea modelului: {accuracy:.2f}')
print(f'Precizia modelului: {precision:.2f}')
