# set up working directory as the location of this script
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# check if R package "mixtools" is installed
if("mixtools" %in% rownames(installed.packages()) == FALSE) {install.packages("xtable")}
require("mixtools")

# receive arguments from command line
args <- commandArgs(trailingOnly = TRUE)
fname <- args[1]
noise_type <- args[2]

# process arguments
if (length(args) < 2){
  # set up default for test run
  fname <- "lin_0_1_2_1_30percent"
  noise_type <- "hetero_error"
}

if(noise_type == "hetero_error"){
  arbvar_sign <- TRUE
}else if(noise_type == "homo_error"){
  arbvar_sign <- FALSE
}else {
  stop("Unknown noise types!")
}

#---------------------------------------------------------------------------#
# read simulated data from file
dir <- file.path("./../data", fname)
X <- as.matrix(read.csv(file = file.path(dir, "X.csv"), header = FALSE))
y <- c(as.matrix(read.csv(file = file.path(dir, "y.csv"), header = FALSE)))
k <- as.numeric(read.csv(file = file.path(dir, "num_component.csv"), header = FALSE))
alpha_true <- c(as.matrix(read.csv(file = file.path(dir, "alpha_true.csv"), header = FALSE)))
B_true <- as.matrix(read.csv(file = file.path(dir, "B_true.csv"), header = FALSE))
sigma_true <- head(c(as.matrix(read.csv(file = file.path(dir, "sigma_true.csv"), header = FALSE))),k)
#---------------------------------------------------------------------------#



#---------------------------------------------------------------------------#
# call regmixEM from R Package mixtools
# see documents on https://rdrr.io/cran/mixtools/man/regmixEM.html#heading-0
#---------------------------------------------------------------------------#
reg_outcome <- regmixEM(y, X, lambda = alpha_true, k = k,
                 beta = B_true, sigma = sigma_true,
                 epsilon = 1e-08,
                 addintercept = FALSE, 
                 arbmean = TRUE, arbvar = FALSE,
                 verb = TRUE)
#---------------------------------------------------------------------------#

# storage estimation results to files
alpha_EM <- reg_outcome$lambda
write.table(alpha_EM, file = file.path(dir, "alpha_EM.csv"),row.names = FALSE, col.names  = FALSE)

B_EM <- reg_outcome$beta
write.table(B_EM, file = file.path(dir, "B_EM.csv"),row.names = FALSE, col.names  = FALSE)

sigma_EM <- reg_outcome$sigma
write.table(sigma_EM, file = file.path(dir, "sigma_EM.csv"),row.names = FALSE, col.names  = FALSE)

#---------------------------------------------------------------------------#