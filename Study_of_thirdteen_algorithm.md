Gaussian Naive Bayes (GNB)

  Bayes' theorem:条件付き確率に関して成り立つ定理
  
                  P(B|A)=P(A|B)P(B)/P(A)->P(B|A1,A2...,An)=P(B)P(A1,...,An|B)/P(A1,...,An)
                  
  Gaussian Naive Bayes->
  
    P(xi|y)=1*exp(-(xi-miu(y))^2/2sigma(y)^2)/root(2pi*sigma(y)^2)（正規分布の連続密度分布）
  
  Used in classification

Bernoulli Naive Bayes (BNB)
  
  There may be multiple features but each one is assumed to be a binary-valued (Bernoulli, boolean) variable.If handed any other kind of data, a BernoulliNB instance may binarize its input.
  
  BenoulliNB might perform better better on some datasets, especially those with shorter documents.

Multinomial Naive Bayes (MNB)
  
  Used in text classification
  
  尤度：ある前提条件に従って結果が出現する場合に、逆に観察結果からみて前提条件が「何々であった」と推測する尤もらしさ（もっともらしさ）を表す数値を、「何々」を変数とする関数として捉えたものである。
  
  最尤法->確率分布fDと分布の母数sitaののわかっている離散確率分布Dが与えられたとして、そこからN個の標本X1,X2...Xnを取り出すことを考えよう。すると分布関数から、観察されたデータが得られる確率を次のようにに計算することができる。
  
  例：箱の中から適当に1つ選んだコインを80回投げ、x1=H,x2=T,...x80=Tのようにサンプリングし、表(H)の観察された回数を数えたところ、表(H)が49回、裏(T)が31回であった,箱に入っているコインの数は無限であると仮定する。それぞれがすべての可能な0<=p<=1の値をとるとする。するとすべての可能な0<=p<=1の値に対して次の尤度関数を最大化しなければならない
  
  L(p)=fd(observe49 Heads out of 80|p)=C(80 49)p^49(1-p)^31
  
  微分
  
  0=d(C(80 49)p^49(1-p)^31)/dp->49p^48(1-p)^31-31p^49(1-p)^30
   =p^48(1-p)^30[49(1-p)-31p]
  
  尤度を最大化するのは明らかにp=49/80

Logistic Regression (LR)
  
  linear modell for classification rather than regression
  
  This implementation can fit binary, One-vs- Rest, or multinomial logistic regression with optional L2 or L1 regularization.
  
  logit(pi)=ln(pi/1-pi)=alph+beta1x1,i+...+betakxk,i, i=1...n
  
  pi=E(Y|Xi)=Pr(Yi=1)
  
  pi=Pr(Yi=1|X)=1/1+e^(-(alph+beta1x1,i+...+betakxk,i))

Stochastic Gradient Descent (SGD)
  
  Stochastic gradient descent is a simple yet very efficient approach to fit linear models.
  
  mean_squared_loss:Using the MSE loss makes sense if the assumption that your outputs are a real-valued function of your inputs, with a certain amount of irreducible Gaussian noise, with constant mean and variance. If these assumptions don’t hold true (such as in the context of classification), the MSE loss may not be the best bet.
  
  huber:二乗誤差損失よりも外れ値に敏感ではない
  
  cross-entropy:binary classification, and the cross-entropy loss provides us with faster learning when our predictions differ significantly from our labels, as is generally the case during the first several iterations of model training.

Passive Aggressive Classifier (PAC)
  
  The passive-aggressive algorithms are a family of algorithms for large-scale learning. They are similar to the Perceptron in that they do not require a learning rate. However, contrary to the Perceptron, they include a regularization parameter C.For classification, PassiveAggressiveClassifier can be used with loss='hinge' (PA-I) or loss='squared_hinge' (PA-II). For regression, PassiveAggressiveRegressor can be used with loss='epsilon_insensitive' (PA-I) or loss='squared_epsilon_insensitive' (PA-II).

Support Vector Classifier (SVC)
  
  Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.
   
  The advantages of support vector machines are:

  *Effective in high dimensional spaces.
  
  *Still effective in cases where number of dimensions is greater than the number of samples.
  
  *Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

  *Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

  The disadvantages of support vector machines include:

  *If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
  
  *SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).
  
  

K-Nearest Neighbor (KNN)

Decision Tree (DT)

Random Forest (RF)

Extra Trees Classifier (ERF)

AdaBoost (AB)

Gradient Tree Boosting (GTB)
