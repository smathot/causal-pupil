library(lmerTest)
df <- read.csv('output/behavior.csv')
attach(df)
lm <- glmer(accuracy ~ valid * inducer + (1+valid*inducer|subject_nr),
            family='binomial')
summary(lm)
lm <- lmer(response_time ~ valid * inducer + (1+valid*inducer|subject_nr))
summary(lm)
