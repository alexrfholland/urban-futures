# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(viridis)

# Set site name
site_name <- "trimmed-parade"

# Read the capability data
file_path <- paste0(site_name, "_capabilities_by_timestep.csv")
capability_data <- read_csv(file_path, show_col_types = FALSE)

# Convert year columns to long format
capability_long <- capability_data %>%
  pivot_longer(
    cols = matches("^\\d+$"),
    names_to = "Year",
    values_to = "Count"
  ) %>%
  mutate(Year = as.numeric(Year))

# Create a directory for plots if it doesn't exist
dir.create("plots", showWarnings = FALSE)

# Print the data structure to verify
print(head(capability_long))

# Create a mapping from CapabilityNo to actual capability names
capability_labels <- capability_long %>%
  select(Persona, Capability, CapabilityNo) %>%
  distinct() %>%
  arrange(Persona, CapabilityNo)

# Create a combined plot with all personas, using CapabilityNo for faceting
p_all <- ggplot(capability_long, aes(x = Year, y = Count, color = NumericIndicator, group = NumericIndicator)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  facet_grid(Persona ~ CapabilityNo, scales = "free") +
  scale_color_viridis_d() +
  scale_x_continuous(breaks = sort(unique(capability_long$Year))) +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Capability Indicators Over Time",
    subtitle = site_name,
    x = "Year",
    y = "Count",
    color = "Numeric Indicator"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    strip.text = element_text(size = 12, face = "bold"),
    panel.grid.minor = element_blank(),
    panel.spacing = unit(1, "lines")
  )

# Print the plot
print(p_all)

# Save the combined plot
ggsave(
  filename = paste0("plots/", site_name, "_all_capabilities.png"),
  plot = p_all,
  width = 15,
  height = 10,
  dpi = 300
)

# Create individual plots for each persona
for (persona in unique(capability_long$Persona)) {
  # Filter data for this persona
  persona_data <- capability_long %>% 
    filter(Persona == persona)
  
  # Create the plot
  p <- ggplot(persona_data, aes(x = Year, y = Count, color = NumericIndicator, group = NumericIndicator)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    facet_wrap(~ Capability, scales = "free", ncol = 3) +
    scale_color_viridis_d() +
    scale_x_continuous(breaks = sort(unique(persona_data$Year))) +
    scale_y_continuous(labels = scales::comma) +
    labs(
      title = paste(persona, "Capability Indicators Over Time"),
      subtitle = site_name,
      x = "Year",
      y = "Count",
      color = "Numeric Indicator"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 10),
      strip.text = element_text(size = 12, face = "bold"),
      panel.grid.minor = element_blank(),
      panel.spacing = unit(1, "lines")
    )
  
  # Save the plot
  ggsave(
    filename = paste0("plots/", site_name, "_", tolower(persona), "_capabilities.png"),
    plot = p,
    width = 12,
    height = 6,
    dpi = 300
  )
}

# Print a message
cat("Capability plots created successfully in the 'plots' directory.\n") 