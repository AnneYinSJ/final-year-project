require(RMOA)
hdt <- HoeffdingTree(numericEstimator = "GaussianNumericAttributeClassObserver")
#data(iris)
#iris <- factorise(iris)
#irisdatastream <- datastream_dataframe(data=iris)
#mymodel <- trainMOA(model = hdt, 
#                    formula = Species ~ Sepal.Length + Sepal.Width + Petal.Length, 
#                    data = irisdatastream)
#scores <- predict(mymodel, newdata=iris, type="response")
pokertraining <- read.csv("C:/Users/HP-PC/Desktop/final-year-project/data/poker-hand-testing.data")
colnames(pokertraining) <- c("Suit1","Card1","Suit2","Card2",
                             "Suit3","Card3","Suit4","Card4",
                             "Suit5","Card5","PokerHand")
pokertraining <- factorise(pokertraining)
pokerstream <- datastream_dataframe(data = pokertraining)
vfdt <- trainMOA(model = hdt, formula = PokerHand ~ Suit1+Card1+Suit2+Card2+Suit3+
                   Card3+Suit4+Card4+Suit5+Card5, data = pokerstream)
pokertest <- read.csv("C:/Users/HP-PC/Desktop/final-year-project/data/poker-hand-training-true.data")
colnames(pokertest) <- c("Suit1","Card1","Suit2","Card2",
                             "Suit3","Card3","Suit4","Card4",
                             "Suit5","Card5","PokerHand")
pokertest <- factorise(pokertest)
scores <- predict(vfdt, newdata = pokertraining, type ="response")