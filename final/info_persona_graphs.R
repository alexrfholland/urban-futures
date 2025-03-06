# info_persona_graphs.R
# Script to create capability graphs using ggplot2

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)  # for comma formatting and log scales
library(stringr)  # for string manipulation

# Set the site name
site_name <- "trimmed-parade"

# Load the data using the exact working path
capabilities_count <- read.csv("data/trimmed_parade_total_resources_per_scenario_timestep.csv")

# Print the first few rows and column names to understand the structure
cat("Column names:", paste(colnames(capabilities_count), collapse=", "), "\n")
print(head(capabilities_count))

# Assuming the data has columns for Persona, Capability, and years as columns
# Convert to long format for plotting
capabilities_long <- capabilities_count %>%
  pivot_longer(cols = matches("^\\d+$"), 
               names_to = "Year", 
               values_to = "Count") %>%
  # Convert Year to numeric
  mutate(Year = as.numeric(as.character(Year)))

# Create a more readable version of capability names
capabilities_long <- capabilities_long %>%
  mutate(
    Capability_Label = str_to_title(str_replace_all(Capability, "_", " "))
  )

# Create faceted plot by persona
p_faceted <- ggplot(capabilities_long, aes(x = Year, y = Count, color = Capability_Label)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_y_continuous(labels = comma) +
  facet_grid(Persona ~ ., scales = "free_y") +
  labs(
    title = "Capabilities by Persona Over Time",
    x = "Year",
    y = "Count",
    color = "Capability"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.position = "bottom",
    strip.text = element_text(size = 12, face = "bold"),
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_line(color = "gray95"),
    panel.spacing = unit(1, "lines")
  )

# Create output directory if it doesn't exist
output_dir <- "plots"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Save the faceted plot
output_file <- file.path(output_dir, paste0(site_name, "_capabilities_by_persona.png"))
ggsave(output_file, p_faceted, width = 15, height = 10, dpi = 300)
cat("Saved faceted plot to:", output_file, "\n")

# Create faceted plot by capability
p_capability <- ggplot(capabilities_long, aes(x = Year, y = Count, color = Persona)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_y_continuous(labels = comma) +
  facet_wrap(~ Capability_Label, scales = "free_y", ncol = 3) +
  labs(
    title = "Capabilities Across Personas Over Time",
    x = "Year",
    y = "Count",
    color = "Persona"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.position = "bottom",
    strip.text = element_text(size = 12, face = "bold"),
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_line(color = "gray95"),
    panel.spacing = unit(1, "lines")
  )

# Save the capability plot
output_file <- file.path(output_dir, paste0(site_name, "_capabilities_by_type.png"))
ggsave(output_file, p_capability, width = 15, height = 10, dpi = 300)
cat("Saved capability plot to:", output_file, "\n")

cat("\nAll plots generated successfully!\n") 