import pandas as pd

df = pd.read_csv('p3.csv')
print(df.columns)
"""
print("Columnas:")

print("Descripcion")
print(df.describe())
print("Info")
print(df.info)
print("Head")
print(df.head())
print("Tail")
print(df.tail())
print("Valores nulos")
print(df.isnull().sum())
"""
#print("Columnas claves")

#columnasKey = df[['Source','Protocol','Info','delta time','Length','Time','Destination']]
#print("Filtrado")
#dnsFiltrado = columnasKey[columnasKey['Protocol']=='DNS'].copy()
#dnsFiltrado = columnasKey['Protocol']=='DNS'
#print(columnasKey['Protocol'].unique())
#print(dnsFiltrado.head(16))
protocoloDNS = df[df['Protocol'] == 'DNS'].copy()
features = protocoloDNS.groupby('Source').agg({
        'No.': 'count',              # Número de consultas DNS por IP
        'Length': ['mean', 'sum'],    # Tamaño promedio y total de bytes
        'delta_time': ['mean', 'std'] # Tiempo entre consultas (con espacio en el nombre)
    }).reset_index()