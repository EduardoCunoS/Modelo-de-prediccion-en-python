import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import tkinter as tk
from tkinter import messagebox

def analizar_prediccion():
    try:
        id_equipo_1 = entry_id_equipo_1.get().strip()
        id_equipo_2 = entry_id_equipo_2.get().strip()

        if not id_equipo_1 or not id_equipo_2:
            messagebox.showerror("Error", "Debes ingresar ambos IDs de equipo.")
            return

        tipos_prediccion = ["Goles", "Remates", "Corners"]
        columnas = {"Goles": "goles", "Remates": "remates", "Corners": "corners"}

        conexion = sqlite3.connect("N_Soccer.db")
        cursor = conexion.cursor()

        fig, axs = plt.subplots(2, 3, figsize=(18, 8))
        fig.suptitle(f'Comparación entre {id_equipo_1} y {id_equipo_2}', fontsize=16)

        resumen = ""

        for col_index, tipo in enumerate(tipos_prediccion):
            columna = columnas[tipo]

            def obtener_datos(id_equipo):
                cursor.execute(f""" 
                    SELECT id_partidos, {columna}
                    FROM partido
                    WHERE id_equipo = ?
                    ORDER BY id_partidos
                """, (id_equipo,))
                return cursor.fetchall()

            datos_1 = obtener_datos(id_equipo_1)
            datos_2 = obtener_datos(id_equipo_2)

            if not datos_1 or not datos_2:
                messagebox.showerror("Error", f"No hay datos suficientes para {tipo}")
                return

            partidos_1 = np.array([fila[0] for fila in datos_1]).reshape(-1, 1)
            valores_1 = np.array([fila[1] for fila in datos_1])
            partidos_2 = np.array([fila[0] for fila in datos_2]).reshape(-1, 1)
            valores_2 = np.array([fila[1] for fila in datos_2])

            if len(partidos_1) < 2 or len(partidos_2) < 2:
                messagebox.showerror("Error", f"Se necesitan al menos 2 partidos para {tipo}")
                return

            scaler_1 = MinMaxScaler()
            valores_1_norm = scaler_1.fit_transform(valores_1.reshape(-1, 1)).flatten()
            scaler_2 = MinMaxScaler()
            valores_2_norm = scaler_2.fit_transform(valores_2.reshape(-1, 1)).flatten()

            poly = PolynomialFeatures(degree=2)
            partidos_1_poly = poly.fit_transform(partidos_1)
            partidos_2_poly = poly.fit_transform(partidos_2)

            modelo_1 = LinearRegression().fit(partidos_1_poly, valores_1_norm)
            modelo_2 = LinearRegression().fit(partidos_2_poly, valores_2_norm)

            valores_1_reales = scaler_1.inverse_transform(valores_1_norm.reshape(-1, 1)).flatten()
            valores_2_reales = scaler_2.inverse_transform(valores_2_norm.reshape(-1, 1)).flatten()
            pred_1 = scaler_1.inverse_transform(modelo_1.predict(partidos_1_poly).reshape(-1, 1)).flatten()
            pred_2 = scaler_2.inverse_transform(modelo_2.predict(partidos_2_poly).reshape(-1, 1)).flatten()

            partido_futuro = np.array([[max(partidos_1[-1][0], partidos_2[-1][0]) + 1]])
            partido_futuro_poly = poly.transform(partido_futuro)
            p1 = scaler_1.inverse_transform(modelo_1.predict(partido_futuro_poly).reshape(-1, 1))[0][0]
            p2 = scaler_2.inverse_transform(modelo_2.predict(partido_futuro_poly).reshape(-1, 1))[0][0]

            resumen += f"{tipo}:\n  {id_equipo_1}: {p1:.2f}\n  {id_equipo_2}: {p2:.2f}\n\n"

            # --- Gráfica para equipo 1 ---
            ax1 = axs[0, col_index]
            ax1.scatter(partidos_1, valores_1_reales, color='blue', label='Datos reales' if col_index == 0 else "")
            ax1.plot(partidos_1, pred_1, color='blue', linestyle='--', label='Tendencia' if col_index == 0 else "")
            ax1.scatter(partido_futuro, p1, color='red', marker='X', s=100, label='Predicción' if col_index == 0 else "")
            ax1.set_title(f"{id_equipo_1} - {tipo}", fontsize=12, fontweight='bold')
            ax1.set_xlabel('Partido', fontsize=10)
            ax1.set_ylabel(tipo, fontsize=10)
            ax1.grid(True)
            if col_index == 0: ax1.legend()

            # --- Gráfica para equipo 2 ---
            ax2 = axs[1, col_index]
            ax2.scatter(partidos_2, valores_2_reales, color='orange', label='Datos reales' if col_index == 0 else "")
            ax2.plot(partidos_2, pred_2, color='orange', linestyle='--', label='Tendencia' if col_index == 0 else "")
            ax2.scatter(partido_futuro, p2, color='red', marker='X', s=100, label='Predicción' if col_index == 0 else "")
            ax2.set_title(f"{id_equipo_2} - {tipo}", fontsize=12, fontweight='bold')
            ax2.set_xlabel('Partido', fontsize=10)
            ax2.set_ylabel(tipo, fontsize=10)
            ax2.grid(True)
            if col_index == 0: ax2.legend()

        conexion.close()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        messagebox.showinfo("Predicciones", resumen)

    except Exception as e:
        messagebox.showerror("Error", f"Hubo un error: {e}")

# Interfaz gráfica mejorada
root = tk.Tk()
root.title("Comparador de Equipos de Fútbol")
root.geometry("400x400")
root.config(bg="#f0f0f0")

# Título de la ventana
title_label = tk.Label(root, text="Comparador de Equipos de Fútbol", font=("Arial", 16, 'bold'), bg="#f0f0f0", fg="#2f4f4f")
title_label.pack(pady=20)

# Etiqueta y campo de texto para ID del primer equipo
label_id_equipo_1 = tk.Label(root, text="ID del primer equipo:", font=("Arial", 12), bg="#f0f0f0")
label_id_equipo_1.pack(pady=5)
entry_id_equipo_1 = tk.Entry(root, font=("Arial", 12), width=20, relief="solid", borderwidth=2)
entry_id_equipo_1.pack(pady=5)

# Etiqueta y campo de texto para ID del segundo equipo
label_id_equipo_2 = tk.Label(root, text="ID del segundo equipo:", font=("Arial", 12), bg="#f0f0f0")
label_id_equipo_2.pack(pady=5)
entry_id_equipo_2 = tk.Entry(root, font=("Arial", 12), width=20, relief="solid", borderwidth=2)
entry_id_equipo_2.pack(pady=5)

# Botón para generar la comparación
btn_analizar = tk.Button(root, text="Generar Comparación", font=("Arial", 12), bg="#4CAF50", fg="white", relief="raised", command=analizar_prediccion)
btn_analizar.pack(pady=20)

root.mainloop()