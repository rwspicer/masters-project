# method steps

1. create statistic maps (mean, std. dev.) for desired variables.
   * i.e FDD (year N), Early Winter Precip [EWP] ( oct - nov year N), 
   Winter Precip[FWP] (Oct year N - Mar year N+1), TDD (year N+1)
   * note std. dev. is not used in this method.

2. For each variable for each year create a '% difference' maps from the mean.
   * ```((current_value - mean) \ |mean|) * 100```

3. for each year take the average of the '% difference' maps for each variable being used to calculate potential initiation areas 
   * I.e ``` (FDD% + EWP% + TDD%) / 3``` or 
   * ``` (FDD% + EWP% + FWP% + TDD%) / 4```


# variations? (not implemented)

* all pixels less than average in '% difference' maps treated a 0 and not negative percents? 
