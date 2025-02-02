library(tidyverse)
library(mgcv)

args <- commandArgs(trailingOnly = TRUE)

data_path <- args[1] #"~/Downloads/Procal/v2/lumos_offsets_quad.tsv"
out_path <- args[2] #"~/Downloads/QE_to_lumos_poly_quads.tsv"
out_path_pdf <- args[3]

data <- read_delim(data_path, delim=" ") %>%
  filter(SA > 0)

spline_x <- seq(5,40,0.01)
coef <- tibble()
for (seq in unique(data$pep)) {
  data_pep <- data %>% filter(pep==seq)
  for (pep_z in unique(data_pep$z))
  {
    data_pep_z <- data_pep %>% filter(z == pep_z)

    fit <- glm(NCE_Lumos ~ poly(NCE_QE, 3, raw=T), data=data.frame(data_pep_z), weights=SA)
    pred <- predict.glm(fit, data.frame(NCE_QE=spline_x))
    spline_data <- tibble(NCE_Lumos=pred, NCE_QE=spline_x)

    p <- (ggplot(data_pep_z, aes(y=NCE_Lumos, x=NCE_QE))
          + geom_line(data=spline_data, linewidth=0.5, show.legend = F)
          + geom_point(show.legend = F)
          + ggtitle(str_c(seq, pep_z))
    )

    ggsave(str_c(out_path_pdf, seq, pep_z, "_lumos2qe.pdf"), p)

    coef <- rbind(coef, tibble(pep = seq, z = pep_z, c0=fit$coefficients[1], c1=fit$coefficients[2], c2=fit$coefficients[3], c3=fit$coefficients[4]))
  }
}

write_tsv(coef, out_path)

