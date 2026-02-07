import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def cargar_datos(archivo_csv):
    """Carga el archivo CSV con punto y coma como delimitador"""
    # Intentar primero con punto y coma
    try:
        df = pd.read_csv(archivo_csv, sep=';', encoding='utf-8')
        print(f"✅ CSV cargado con delimitador ';'")
        return df
    except:
        # Si falla, intentar con coma
        df = pd.read_csv(archivo_csv, encoding='utf-8')
        print(f"✅ CSV cargado con delimitador ','")
        return df

def mostrar_columnas(df):
    """Muestra las columnas disponibles en el CSV"""
    print("\n📋 Columnas encontradas en el CSV:")
    print("-" * 80)
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. '{col}'")
    print("-" * 80)

def limpiar_valores(valor):
    """Limpia valores que vienen entre comillas"""
    if isinstance(valor, str):
        return valor.strip('"').strip()
    return valor

def normalizar_columnas(df):
    """Normaliza los nombres de las columnas y limpia los datos"""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Limpiar valores de todas las columnas
    for col in df.columns:
        df[col] = df[col].apply(limpiar_valores)
    
    return df

def calcular_metricas(df):
    """Calcula las métricas de rendimiento para cada trabajador"""
    
    resultados = []
    
    for idx, row in df.iterrows():
        try:
            # Extraer datos básicos
            rol = str(row.get('rol', 'N/A'))
            
            # Convertir valores numéricos limpiando comillas
            count = float(str(row.get('count', 0)).strip('"'))
            cantidad_por_minuto = float(str(row.get('cantidadporminuto', 0)).strip('"'))
            
            # Sumar todas las cantidades por hora (9-19)
            cantidad_9a19 = 0
            for i in range(9, 19):
                col_name = f'cantidad_{i}a{i+1}'
                if col_name in df.columns:
                    valor = str(row.get(col_name, 0)).strip('"')
                    cantidad_9a19 += float(valor) if valor else 0
            
            # Calcular promedio por hora (todas las horas con datos)
            total_por_horas = 0
            horas_contadas = 0
            
            for i in range(9, 19):
                col_name = f'cantidad_{i}a{i+1}'
                if col_name in df.columns:
                    valor = str(row.get(col_name, 0)).strip('"')
                    valor_num = float(valor) if valor else 0
                    if valor_num > 0:
                        total_por_horas += valor_num
                        horas_contadas += 1
            
            promedio_por_hora = total_por_horas / horas_contadas if horas_contadas > 0 else 0
            
            # Calcular score de productividad
            score_productividad = (
                (cantidad_por_minuto * 100) + 
                (cantidad_9a19 * 0.5) + 
                (promedio_por_hora * 2)
            )
            
            # Calcular eficiencia (% de trabajo en horario laboral)
            eficiencia = (cantidad_9a19 / count * 100) if count > 0 else 0
            
            resultados.append({
                'Rol': rol,
                'Usuario': str(row.get('usu_usu', 'N/A')),
                'Total_Registros': int(count),
                'Por_Minuto': round(cantidad_por_minuto, 2),
                'Trabajo_9a19': int(cantidad_9a19),
                'Promedio_Por_Hora': round(promedio_por_hora, 2),
                'Score_Productividad': round(score_productividad, 2),
                'Eficiencia_%': round(eficiencia, 2)
            })
        except Exception as e:
            print(f"⚠️ Error procesando fila {idx}: {str(e)}")
            continue
    
    return pd.DataFrame(resultados)

def clasificar_trabajadores(df_metricas):
    """Clasifica a los trabajadores en 3 categorías"""
    
    if len(df_metricas) == 0:
        print("❌ No hay datos para clasificar")
        return None, None, None, None
    
    # Ordenar por score de productividad
    df_sorted = df_metricas.sort_values('Score_Productividad', ascending=False)
    
    total = len(df_sorted)
    top_30 = max(1, int(total * 0.3))
    top_70 = max(2, int(total * 0.7))
    
    # Clasificar
    top_performers = df_sorted.iloc[:top_30].copy()
    rendimiento_medio = df_sorted.iloc[top_30:top_70].copy()
    necesitan_mejora = df_sorted.iloc[top_70:].copy()
    
    top_performers['Categoria'] = '🏆 Top Performers'
    rendimiento_medio['Categoria'] = '📊 Rendimiento Medio'
    necesitan_mejora['Categoria'] = '⚠️ Necesitan Mejora'
    
    return top_performers, rendimiento_medio, necesitan_mejora, df_sorted

def generar_reporte(top, medio, bajo, df_completo):
    """Genera un reporte completo en consola"""
    
    print("\n" + "="*80)
    print(" 📊 ANÁLISIS DE RENDIMIENTO LABORAL ".center(80, "="))
    print("="*80 + "\n")
    
    # Estadísticas generales
    print("📈 ESTADÍSTICAS GENERALES")
    print("-" * 80)
    print(f"Total de trabajadores: {len(df_completo)}")
    print(f"Score promedio: {df_completo['Score_Productividad'].mean():.2f}")
    print(f"Score máximo: {df_completo['Score_Productividad'].max():.2f}")
    print(f"Score mínimo: {df_completo['Score_Productividad'].min():.2f}")
    print(f"Eficiencia promedio: {df_completo['Eficiencia_%'].mean():.2f}%")
    
    # Top Performers
    print("\n" + "="*80)
    print(f"🏆 TOP PERFORMERS ({len(top)} trabajadores - Top 30%)")
    print("="*80)
    print(top.to_string(index=False))
    
    # Rendimiento Medio
    if len(medio) > 0:
        print("\n" + "="*80)
        print(f"📊 RENDIMIENTO MEDIO ({len(medio)} trabajadores - 30-70%)")
        print("="*80)
        print(medio.to_string(index=False))
    
    # Necesitan Mejora
    if len(bajo) > 0:
        print("\n" + "="*80)
        print(f"⚠️ NECESITAN MEJORA ({len(bajo)} trabajadores - Bottom 30%)")
        print("="*80)
        print(bajo.to_string(index=False))
    
    print("\n" + "="*80 + "\n")

def crear_visualizaciones(top, medio, bajo, df_completo):
    """Crea visualizaciones gráficas"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Gráfico de pastel - Distribución por categoría
    ax1 = plt.subplot(2, 3, 1)
    categorias = ['Top Performers', 'Rendimiento Medio', 'Necesitan Mejora']
    valores = [len(top), len(medio), len(bajo)]
    colores = ['#10b981', '#f59e0b', '#ef4444']
    
    ax1.pie(valores, labels=categorias, autopct='%1.1f%%', colors=colores, startangle=90)
    ax1.set_title('Distribución por Categoría', fontsize=14, fontweight='bold')
    
    # 2. Top 10 - Score de Productividad
    ax2 = plt.subplot(2, 3, 2)
    top_n = min(10, len(df_completo))
    top_10 = df_completo.nlargest(top_n, 'Score_Productividad')
    ax2.barh(range(len(top_10)), top_10['Score_Productividad'], color='#6366f1')
    ax2.set_yticks(range(len(top_10)))
    ax2.set_yticklabels(top_10['Usuario'], fontsize=8)
    ax2.set_xlabel('Score de Productividad')
    ax2.set_title(f'Top {top_n} Trabajadores', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    
    # 3. Bottom 10 - Score de Productividad
    ax3 = plt.subplot(2, 3, 3)
    bottom_n = min(10, len(df_completo))
    bottom_10 = df_completo.nsmallest(bottom_n, 'Score_Productividad')
    ax3.barh(range(len(bottom_10)), bottom_10['Score_Productividad'], color='#ef4444')
    ax3.set_yticks(range(len(bottom_10)))
    ax3.set_yticklabels(bottom_10['Usuario'], fontsize=8)
    ax3.set_xlabel('Score de Productividad')
    ax3.set_title(f'Bottom {bottom_n} Trabajadores', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    
    # 4. Distribución de Score de Productividad
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(df_completo['Score_Productividad'], bins=min(20, len(df_completo)), 
             color='#6366f1', alpha=0.7, edgecolor='black')
    ax4.axvline(df_completo['Score_Productividad'].mean(), color='red', 
                linestyle='--', linewidth=2, label='Promedio')
    ax4.set_xlabel('Score de Productividad')
    ax4.set_ylabel('Frecuencia')
    ax4.set_title('Distribución de Scores', fontsize=14, fontweight='bold')
    ax4.legend()
    
    # 5. Eficiencia vs Total de Registros
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(df_completo['Total_Registros'], df_completo['Eficiencia_%'], 
                         c=df_completo['Score_Productividad'], cmap='viridis', s=100, alpha=0.6)
    ax5.set_xlabel('Total de Registros')
    ax5.set_ylabel('Eficiencia (%)')
    ax5.set_title('Eficiencia vs Volumen de Trabajo', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax5, label='Score')
    
    # 6. Comparación por categoría - Boxplot
    ax6 = plt.subplot(2, 3, 6)
    todos = pd.concat([top, medio, bajo])
    if len(todos) > 0:
        sns.boxplot(data=todos, x='Categoria', y='Score_Productividad', ax=ax6, palette=colores)
        ax6.set_xlabel('')
        ax6.set_ylabel('Score de Productividad')
        ax6.set_title('Comparación entre Categorías', fontsize=14, fontweight='bold')
        ax6.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig('analisis_rendimiento.png', dpi=300, bbox_inches='tight')
    print("✅ Gráficos guardados en 'analisis_rendimiento.png'")
    plt.show()

def exportar_resultados(top, medio, bajo, df_completo):
    """Exporta los resultados a archivos Excel"""
    
    try:
        with pd.ExcelWriter('analisis_rendimiento_completo.xlsx', engine='openpyxl') as writer:
            df_completo.to_excel(writer, sheet_name='Todos', index=False)
            top.to_excel(writer, sheet_name='Top Performers', index=False)
            if len(medio) > 0:
                medio.to_excel(writer, sheet_name='Rendimiento Medio', index=False)
            if len(bajo) > 0:
                bajo.to_excel(writer, sheet_name='Necesitan Mejora', index=False)
        
        print("✅ Resultados exportados a 'analisis_rendimiento_completo.xlsx'")
    except Exception as e:
        print(f"⚠️ No se pudo exportar a Excel: {str(e)}")

def main():
    """Función principal"""
    
    print("\n🚀 Iniciando análisis de rendimiento laboral...\n")
    
    # Cargar datos
    archivo = input("Ingresa el nombre del archivo CSV (ejemplo: reporte.csv): ").strip()
    
    try:
        df = cargar_datos(archivo)
        print(f"✅ Archivo cargado correctamente: {len(df)} registros encontrados")
        
        # Mostrar columnas disponibles
        mostrar_columnas(df)
        
        # Normalizar nombres de columnas
        df = normalizar_columnas(df)
        
        print("\n📋 Columnas después de normalizar:")
        print(list(df.columns))
        
        # Calcular métricas
        print("\n📊 Calculando métricas de rendimiento...")
        df_metricas = calcular_metricas(df)
        
        if len(df_metricas) == 0:
            print("❌ No se pudieron calcular métricas. Verifica el formato del CSV.")
            return
        
        print(f"✅ Métricas calculadas para {len(df_metricas)} trabajadores")
        
        # Clasificar trabajadores
        print("\n🔍 Clasificando trabajadores...")
        top, medio, bajo, df_completo = clasificar_trabajadores(df_metricas)
        
        if df_completo is None:
            print("❌ No se pudo completar la clasificación")
            return
        
        # Generar reporte en consola
        generar_reporte(top, medio, bajo, df_completo)
        
        # Crear visualizaciones
        print("\n📈 Generando gráficos...")
        crear_visualizaciones(top, medio, bajo, df_completo)
        
        # Exportar a Excel
        print("\n💾 Exportando resultados...")
        exportar_resultados(top, medio, bajo, df_completo)
        
        print("\n✅ ¡Análisis completado exitosamente!")
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{archivo}'")
        print("💡 Asegúrate de que el archivo esté en la misma carpeta que el script")
    except Exception as e:
        print(f"❌ Error al procesar los datos: {str(e)}")
        import traceback
        print("\n🔍 Detalles del error:")
        traceback.print_exc()

if __name__ == "__main__":
    main()