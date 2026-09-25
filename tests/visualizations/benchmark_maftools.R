
# Install BiocManager if not present
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos = "https://cloud.r-project.org")

# Install maftools if not present
if (!require("maftools", quietly = TRUE))
    BiocManager::install("maftools")

library(maftools)

# Create output directories
output_dir <- "tests/visualizations/maftools_benchmark_output"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Define input data path (same as PyMut examples)
maf_file <- "src/pyMut/data/examples/MAF/tcga_laml.maf.gz"

# Function to benchmark and save plot
benchmark_plot <- function(name, plot_func) {
    png_file <- file.path(output_dir, paste0(name, ".png"))
    
    start_time <- Sys.time()
    png(filename = png_file, width = 1000, height = 800)
    tryCatch({
        plot_func()
    }, error = function(e) {
        cat(sprintf("Error in %s: %s\n", name, e$message))
    })
    dev.off()
    end_time <- Sys.time()
    
    duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
    cat(sprintf("%s: %.4f s\n", name, duration))
    return(duration)
}

# Load data once for plotting benchmarks
laml <- read.maf(maf = maf_file, verbose = FALSE)

# 1. Summary Plot
benchmark_plot("summary_plot", function() {
    plotmafSummary(maf = laml, rmOutlier = TRUE, addStat = 'median', dashboard = TRUE, titvRaw = FALSE)
})

# 2. Oncoplot
benchmark_plot("oncoplot", function() {
    oncoplot(maf = laml, top = 10)
})

# 3. Lollipop Plot
benchmark_plot("lollipop_plot", function() {
    lollipopPlot(
        maf = laml,
        gene = 'DNMT3A',
        AACol = 'Protein_Change',
        showMutationRate = TRUE,
        labelPos = 882
    )
})

# 4. Somatic Interactions
benchmark_plot("somatic_interactions", function() {
    somaticInteractions(maf = laml, top = 25, pvalue = c(0.05, 0.1))
})

cat("Benchmark completed.\n")
