library(tidyverse)

args <- commandArgs(trailingOnly = TRUE)
idx = as.integer(args[1])
inpath = args[2]
outpath = args[3]

AA_idx = 1 + ((idx-1) %% 20)
slice_idx = 1 + floor((idx-1) / 20)

AAs = "ACDEFGHIKLMNPQRSTVWY"
AA = str_sub(AAs,AA_idx,AA_idx)

print(idx)
print(AA_idx)
print(slice_idx)
print(AA)

data <- read_tsv(str_c(inpath, "/", AA, ".tsv"))
outfile <- file(str_c(outpath, "/", AA, "_", slice_idx, ".msp"), open = "w")

write_msp <- function(outfile, pep, charge, mod, frags)
  for (NCE_v in unique(frags$NCE)) {
    # Write name      VHNQEEYAR/3_0_NCE22.06_100.0-1158.0_0.0385295
    seq_name = str_c("Name: ", pep, "/", charge)
    NCE_str = str_c("NCE", NCE_v)
    name_str <- str_c(seq_name, mod, NCE_str, "0-2000", "0", sep="_")
    # Write comment
    comment_str <- str_c("Comment: ", "RawOvFtT=1000 Purity=1 NCE_aligned=", NCE_v, " LowMZ=0 HighMZ=2000 LOD=0 IsoWidth=0.1 IsoCenter=0 z=", charge)
    # Write num peaks
    num_peaks_str <- str_c("Num peaks: ", length(unique(frags$frag)))

    writeLines(c(name_str, comment_str, num_peaks_str), outfile)

    # Write peaks     102.0533        0.2606142       ?       0
    frags_NCE <- frags %>% filter(NCE == NCE_v)
    peak_lines = c()
    for (i in 1:nrow(frags_NCE)) {
      peak_line <- str_c(i, max(0, frags_NCE$intensity[i]), frags_NCE$frag[i], "0", sep="\t")
      peak_lines <- append(peak_lines, peak_line)
    }
    writeLines(peak_lines, outfile)

    # Write newline
    writeLines(c(""), outfile)
  }


unique_peps = unique(data$seq)
slice_length = ceiling(length(unique_peps) / 50)
unique_peps = unique_peps[(slice_idx * slice_length) : min(length(unique_peps), (slice_idx+1) * slice_length)]

data <- data %>%
  filter(seq %in% unique_peps) %>%
  group_by(seq, mods, z, frag) %>%
  filter(length(unique(NCE)) > 3) %>%
  group_by(seq, mods, z, frag, NCE) %>%
  summarize(intensity = weighted.mean(intensity, weight))
  #summarize(intensity = median(intensity))



for (pep in unique(data$seq)) {
  data.sub.pep <- data %>% filter(seq == pep)

  for (charge in unique(data.sub.pep$z)) {

    data.sub.z <- data.sub.pep %>% filter(z == charge)

    for (mod in unique(data.sub.z$mods)) {

      data.sub <- data.sub.z %>% filter(mods == mod)

      spline_x <- seq(min(data.sub$NCE)-1, max(data.sub$NCE)+1, 0.1)

      # FIT SPLINES
      spline.fits <- data.frame(frag=c(), NCE=c(), intensity=c(), pen=c())
      for (frag_name in unique(data.sub$frag)) {
        data.sub.frag <- data.sub %>% filter(frag==frag_name)

        if (length(unique(data.sub.frag$NCE)) < 4) { next }

        tryCatch(
            {
                fit <- smooth.spline(data.sub.frag$NCE, data.sub.frag$intensity, nknots = length(unique(data.sub.frag$NCE)))

                pred <- data.frame(stats:::predict.smooth.spline(fit, spline_x))

                y <- (data.frame(stats:::predict.smooth.spline(fit, data.sub.frag$NCE))$y - data.sub.frag$intensity) / max( data.sub.frag$intensity)
                err <- sum(y * y) / length(y)

                if (err > 1e-3)  {next}

                spline.fits.frags <- data.frame(frag=frag_name, NCE=pred$x, intensity=pred$y)
                spline.fits <- rbind(spline.fits, spline.fits.frags)
            }, error=function(e){}
       
        )
      }
      write_msp(outfile, pep, charge, mod, spline.fits)
    }
  }
}

close(outfile)
