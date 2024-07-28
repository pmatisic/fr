# Učitavanje potrebnih biblioteka
if(!require(ggplot2)) {
  install.packages("ggplot2")
  library(ggplot2)
}

# Učitavanje podataka
data <- read.csv("/home/pijetao/Preuzimanja/FR/results/dob.csv")

# Prikaz prvih nekoliko redaka
head(data)
summary(data)

# Provjera i ispis redova s neispravnim vrijednostima
sapply(data, function(x) sum(is.na(x)))
table(data$Gender) 

# Histogram uz KDE krivulju
ggplot(data, aes(x=Age)) +
  geom_histogram(aes(y=after_stat(density)), binwidth=1, fill="skyblue", color="black", alpha=0.7) +
  geom_density(color="red") +
  xlim(0, 120) +
  labs(title="Distribution of Ages", x="Age", y="Density")
