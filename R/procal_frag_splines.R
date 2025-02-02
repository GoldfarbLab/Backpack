library(tidyverse)

args <- commandArgs(trailingOnly = TRUE)

data <- read_tsv(args[1]) #"/Users/dennisgoldfarb/Downloads/ProCal/v2/procal_traces_ambig.tsv"
out_path <- args[2] #"/Volumes/d.goldfarb/Active/Projects/Backpack/python/spline_fits_sqrt_quad.tsv"
out_path_pdf <- args[3]

data.avg <- data %>%
  group_by(pep, z, frag, NCE) %>%
  summarize(intensity = weighted.mean(intensity, sqrt(rawOvFtT*purity), na.rm=T)) %>%
  group_by(pep, z, frag) %>%
  filter(sum(intensity > 0, na.rm=T) >= 3, max(intensity, na.rm=T) > 0.01) #%>% # frag seen in at least 3 NCEs

NCEs = c(10,15,20,22,24,26,28,30,32,34,36,38,40,45,50)
spline_x <- seq(10,50,0.1)

all_spline_fits = data.frame(peptide=c(), z=c(), frag=c(), coefficients=c(), knots=c(), pen=c())

for (seq in unique(data.avg$pep)) {

  data.sub.pep <- data.avg %>% filter(pep == seq)

  for (charge in unique(data.sub.pep$z)) {

    data.sub <- data.sub.pep %>%
      filter(z == charge) %>%
      mutate(missing = F)

    data.sub.separate <- data.sub %>%
      mutate(ambig = str_detect(frag, ",")) %>%
      separate_longer_delim(frag, delim=",")

    # FIT SPLINES
    spline.fits <- data.frame(frag=c(), NCE=c(), intensity=c(), pen=c())
    for (frag_name in unique(data.sub$frag)) {

      data.sub.frag <- data.sub %>% filter(frag==frag_name)

      # replace missing values with NAs if found in ambig/clean data
      for (myNCE in NCEs) {
        if (!myNCE %in% unique(data.sub.frag$NCE)) {
          data.sub.separate.frag <- data.sub.separate %>% filter(frag == frag_name, NCE == myNCE)
          if (nrow(data.sub.separate.frag) > 0) {
            data.sub.frag <- rbind(data.sub.frag, data.frame(NCE=myNCE, intensity=NA))
          }
          # this is ambig and we nee to look for clean options
          if (str_detect(frag_name, ",")) {
            individual_frag_names <- unlist(str_split(frag_name, ","))
            for (i_frag_name in individual_frag_names) {
              data.sub.separate.frag <- data.sub.separate %>% filter(frag == i_frag_name, NCE == myNCE)
              if (nrow(data.sub.separate.frag) > 0) {
                data.sub.frag <- rbind(data.sub.frag, data.frame(NCE=myNCE, intensity=NA))
                break
              }
            }
          }
        }
      }

      data.sub.frag <- data.sub.frag %>% arrange(NCE)

      min.NCE <- min(data.sub.frag$NCE)
      max.NCE <- max(data.sub.frag$NCE)

      # Pad with zeros at the ends
      for (myNCE in NCEs) {
        if (!myNCE %in% unique(data.sub.frag$NCE)) {
          if (myNCE < min.NCE || myNCE > max.NCE) {
            data.sub.frag <- rbind(data.sub.frag, data.frame(NCE=myNCE, intensity=0))
            data.sub <- rbind(data.sub, data.frame(pep=seq, z=charge, frag=frag_name, NCE=myNCE, intensity=0, missing=T))
          }
        }
      }

      norm_factor <- -1*max(data.sub.frag$intensity, na.rm=T)/1000
      # Go less than zero to avoid non-zero spline effects
      # find last initial zero - left side
      last_initial_nonzero = 0
      for (row in 1:(nrow(data.sub.frag)-1)) {
        if (!is.na(data.sub.frag[row,]$intensity) && data.sub.frag[row,]$intensity == 0 && !is.na(data.sub.frag[row+1,]$intensity) && data.sub.frag[row+1,]$intensity == 0) {
          last_initial_nonzero <- row+1
        } else {
          break
        }
      }
      if (last_initial_nonzero > 0) {
        for (row in 1:last_initial_nonzero) {
          data.sub.frag[row,]$intensity <- (norm_factor) * ((last_initial_nonzero-row))
        }
      }

      # find first final zero - right side
      first_final_nonzero = nrow(data.sub.frag)
      for (row in nrow(data.sub.frag):2) {
        if (!is.na(data.sub.frag[row,]$intensity) && data.sub.frag[row,]$intensity == 0 && !is.na(data.sub.frag[row-1,]$intensity) && data.sub.frag[row-1,]$intensity == 0) {
          first_final_nonzero <- row-1
        } else {
          break
        }
      }
      if (first_final_nonzero < nrow(data.sub.frag)) {
        for (row in first_final_nonzero:nrow(data.sub.frag)) {
          data.sub.frag[row,]$intensity <- (norm_factor) * ((row-first_final_nonzero))
        }
      }

      data.sub.frag <- data.sub.frag %>% filter(!is.na(intensity))

      if (nrow(data.sub.frag) < 4) { next }

      fit <- smooth.spline(data.sub.frag$NCE, data.sub.frag$intensity, nknots = length(unique(data.sub.frag$NCE)))

      pred <- (data.frame(stats:::predict.smooth.spline(fit, data.sub.frag$NCE))$y - data.sub.frag$intensity) / max( data.sub.frag$intensity)
      err <- sum(pred * pred) / length(pred)

      pred <- data.frame(stats:::predict.smooth.spline(fit, spline_x))
      spline.fits.frags <- data.frame(frag=frag_name, NCE=pred$x, intensity=pred$y, pen=err)
      spline.fits <- rbind(spline.fits, spline.fits.frags)

      knots = (fit$fit$knot * fit$fit$range) + fit$fit$min

      all_spline_fits = rbind(all_spline_fits, data.frame(peptide=c(seq),
                                                          z=c(charge),
                                                          frag=c(frag_name),
                                                          coefficients=paste0(fit$fit$coef, collapse = "", sep=","),
                                                          knots=paste0(knots, collapse = "", sep=","),
                                                          pen=err)
      )
    }

    spline.fits$intensity[spline.fits$intensity < 0] = 0


    # PLOT SPLINES
    p <- (ggplot(data.sub %>% filter(frag %in% spline.fits$frag), aes(x=NCE, y=intensity, group=frag, color=frag))
               + ggtitle(seq)
               + geom_line(data=spline.fits, linewidth=0.5, show.legend = F)
               #+ geom_label(data=spline.fits %>% group_by(frag) %>% summarize(pen=mean(pen)), aes(x=35, y=0, label=scales::scientific(pen, digits = 3)), show.legend = F, fill="#cccccc")
               + geom_point(show.legend = F, aes(shape=factor(missing)))
               + facet_wrap(~ frag, scales='free')
               + scale_x_continuous(breaks=c(10,15,20,24,28,32,36,40,45,50))
               + theme(panel.grid.major = element_line(colour = "#eeeeee"), panel.grid.minor = element_blank(),
                       panel.background = element_blank(), axis.line = element_line(colour = "#333333"))
    )

    ggsave(str_c(out_path_pdf,seq,charge,"_quad_sqrt.pdf"), p)
  }

}

write_delim(all_spline_fits, out_path, delim="\t")
