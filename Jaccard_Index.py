import pandas as pd
from pyMut.input import read_maf
import matplotlib.pyplot as plt
from upsetplot import from_memberships, plot

# --- 1. DATOS Y PROCESAMIENTO (Igual que antes) ---
maf_path = "res/tcga_paad.maf"
py_mut = read_maf(path=maf_path, assembly="37")
df_pymut = py_mut.data

# Filtro maftools
mutations_to_ignore = ['SILENT', 'DE_NOVO_START_OUTOFFRAME', 'DE_NOVO_START_INFRAME',
                       'START_CODON_SNP', 'START_CODON_DEL', 'START_CODON_INS']
df_pymut = df_pymut[~df_pymut['Variant_Classification'].isin(mutations_to_ignore)].copy()

# IDs únicos
df_pymut['unique_id'] = (
    df_pymut['CHROM'].astype(str).str.replace('^chr', '', regex=True) + "_" +
    df_pymut['POS'].astype(str) + "_" +
    df_pymut['Tumor_Sample_Barcode'].astype(str) + "_" +
    df_pymut['Reference_Allele'].astype(str) + "_" +
    df_pymut['Tumor_Seq_Allele2'].astype(str)
)

set_pymut = set(df_pymut['unique_id'])

with open("res/maftools_valid_mutations.txt", "r") as f:
    set_maftools = set(line.strip() for line in f)

intersection = len(set_pymut.intersection(set_maftools))
n_maftools_only = len(set_maftools - set_pymut)
jaccard_val = intersection / len(set_pymut.union(set_maftools))

# --- 2. PREPARACIÓN DEL UPSET PLOT PROFESIONAL ---

data = from_memberships(
    [['maftools'], ['pyMut', 'maftools']],
    data=[n_maftools_only, intersection]
)

# Configuración de estilo "Nature/Science"
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'pdf.fonttype': 42, # Para que el PDF sea editable en Illustrator
    'ps.fonttype': 42
})

fig = plt.figure(figsize=(10, 7))

# Generar el gráfico
# show_counts='{:,}' añade el separador de miles
upset = plot(data, fig=fig, facecolor="#2b7bba", show_counts='{:,}', element_size=None)

# --- 3. LIMPIEZA Y ESTÉTICA ---
for ax_name, ax in upset.items():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=12)
    # Poner etiquetas de ejes en negrita
    ax.set_ylabel(ax.get_ylabel(), fontweight='bold', fontsize=12)
    ax.set_xlabel(ax.get_xlabel(), fontweight='bold', fontsize=12)

# --- 4. POSICIONAR JACCARD EN EL ESPACIO EN BLANCO ---
# Usamos coordenadas relativas de la figura (0,0 es abajo-izq, 1,1 es arriba-der)
# El espacio en blanco superior izquierdo suele estar por (0.1, 0.8)
fig.text(0.18, 0.88, f'Jaccard Index: {jaccard_val:.4f}',
         fontsize=16, fontweight='bold',
         va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

# --- 5. GUARDADO EN ALTA RESOLUCIÓN ---
# El PDF es vectorial (calidad infinita para el paper)
plt.savefig("UpSet_Jaccard_Final.pdf", format='pdf', bbox_inches='tight', transparent=True)
# El PNG a 1200 DPI es ultra-nítido
plt.savefig("UpSet_Jaccard_Final.png", format='png', dpi=1200, bbox_inches='tight', transparent=True)

print(f"Figura generada con éxito. Jaccard: {jaccard_val:.4f}")
plt.show()