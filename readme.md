# Meta Data Selection


Say a batch of data goes through $ f_{select} $ network and predict a batch of $ \alpha $. Those $ \alpha$ represent how representative that data is i.e the possibility of that data being selected as the validation data.

The same batch of data will also goes through the $ f_{seg} $ network that is trained simultaneously and produce a batch of $ \textbf{l} $ which are the dice score of each class. For each data inside that mini-batch, we have a tuple of pairs:
$$ \{ \alpha, \textbf{l} \}_{i} $$

The tuple is the possibility of being selected as the validation data and dice score.

The MMD is the measure of how two distributions are similar to each other. The MMD can measure similarity between predicted validation dice and predicted training dice.

### MMD Loss calculation
Choose the 25% of the tuples that has the max $\alpha$ as the predicted validation dataset, the other 75% of the tuples as the predicted training dataset. We have $I$ number of validation data and $J$ number of training data. If the mini-batch size is $N$, then $N=I+J$.

The original MMD of two distributation is:
$$ \dfrac{   \sum_{i\neq i^{'}}^{I}k(x_{i}, x_{i^{'}})} {I(I-1)}  +    \dfrac{   \sum_{j\neq j^{'}}^{I}k(x_{j}, x_{j^{'}})} {J(J-1)} - \dfrac{   \sum_{i,j=1}^{I, J}k(x_{i}, x_{j})} {IJ}$$

There are three parts in the MMD loss and each part has a sum of kernel function applied to two distributations' sample. To make this function differentialble with represt to $ f_{select} $. We do a weighted average of result function. 

$$ \dfrac{   \sum_{i\neq i^{'}}^{I}  [(1-\alpha_{i})+(1-\alpha_{i^{'}})]   k(x_{i}, x_{i^{'}})} {I(I-1)}  +    \dfrac{   \sum_{j\neq j^{'}}^{I}   [\alpha_{j}+\alpha_{j^{'}}]  k(x_{j}, x_{j^{'}})} {J(J-1)} - \dfrac{   \sum_{i,j=1}^{I, J}  [\alpha_{i}+(1-\alpha_{j})]  k(x_{i}, x_{j})} {IJ}$$
