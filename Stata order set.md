<!-- <style>
@page {
    size: A4;
    margin: 20mm;
}
body {
    font-family: Arial, sans-serif;
    font-size: 14pt; /* 调整全局字体大小 */
    line-height: 1.5;
}
</style> -->

<div align="center" style="font-size:23px;">The empirical method of economic research and stata code</div>

在做任何分析之前都要做协变量平衡分析，防止由于对照组和控制组变量分布造成的误差。

- [1.Random Experiment](#1random-experiment)
- [2.OLS](#2ols)
- [3.Limit dependent varible](#3limit-dependent-varible)
- [4.Matching](#4matching)
- [5.Instrument Variable](#5instrument-variable)
  - [弱工具变量检验](#弱工具变量检验)
  - [外生性（排除性）检验](#外生性排除性检验)
  - [过度识别检验](#过度识别检验)
- [实用小代码stata](#实用小代码stata)
- [一些方法](#一些方法)

## <div style="font-size:25px;">1.Random Experiment</div>

1. 在进行因果估计之前为了避免存在样本分布问题，或者选择性问题，通常会对对照组和样本组进行随机化分析，即计算对照组和实验组具有近似的样本分布。这样可以表示条件独立性。
    
    ```stata
    // 随机实验验证 对于分组进行验证 检查子组内的平衡
    gen subgroup = group(变量) // 生成分组变量   这个公式会生成一个新的变量，这个变量是根据原来的变量进行取分组值的
    bysort subgroup: summarize(变量) // 按照分组变量进行分组，然后对变量进行描述性统计 因为产生的太快了，需要一个变量一个变量跑 ，然后j子组内对照组和实验组进行对比
    ```
    
    - **分组求回归等公式**
    
    ```stata
    // 分组求回归等公式
    bys subgroup: logit/reg y x
    ```











## <div style="font-size:25px;">2.OLS</div>

>异方差指的是误差，由于误差项不确定，所以假设对于每一个i都有一个分布，由$\beta$的推导知异方差的影响，从回归分布图也可以看出来，同方差的分布相对于回归线是均匀的，但是异方差不均匀。（误差由于截距的存在，均值为0）

1. **OLS回归**  

    ```stata
    reg y x1 x2 x3//robust 异方差情况
    ```

2. **加权回归**
    加权回归可以解决异方差问题。
    利用加权对于其残差平方和进行加权，进行最小化.其权重可以选择倾向得分.
    
    ```stata
    reg y x1 x2 x3 [aweight = weight] //加权回归
    ```











## <div style="font-size:25px;">3.Limit dependent varible</div>

***为什么受限被解释变量不能使用OLS：OLS会产生异方差问题，同时会导致预测值大于1或者小于0，这没有意义。***
当相关变量是虚拟变量或选择变量时，我们必须使用其他模型，例如 logit 或probit模型来估计模型

1. **Logit模型**  

    ```stata
    logit y x1 x2 x3 //默认使用最大似然估计
    //关于logit的迭代(optimal函数的要求)以及公式可以看崔学彬的ppt，就是MLE和回归的替换
    logit y x1 x2 x3, or //odds ratio输出就是 $exp(\beta)$
    //由于我们只能通过Odds变化的倍数推断出概率的变化方向，为了推断实际概率。用边际处理
    //利用logit求平均处理效应
    margins, dydx(x1) //其求x1对因变量的平均处理效应，系数为概率变化值（百分比衡量）
    //当 x1增加 1 个单位时，y=1的概率变化的百分比
    margins, dydx(x1) at(x1=0) //求x1=0时的平均处理效应，其他值为均值
    margins, dydx(x1) atmeans //求均值时的平均处理效应
    ```

<div style="color:blue;"><b>logit模型使用logit函数，而probit使用逆正态函数函数</b></div>  

2. **Probit模型**  

    ```stata
    probit y x1 x2 x3 //默认使用最大似然估计
    //由于无法使用probit模型求解odds，只能使用边际处理
    margins, dydx(x1) //其求x1对因变量的平均处理效应，系数为概率变化值（百分比衡量）
    //当 x1增加 1 个单位时，y=1的概率变化的百分比
    margins, dydx(x1) at(x1=0) //求x1=0时的平均处理效应
    margins, dydx(x1) atmeans //求均值时的平均处理效应
    ```

3. **泊松分布**
    条件1：一个事件的发生不影响其它事件的发生，即事件独立发生，不存在传染性、聚集性的事件。
    条件2：因变量Y服从Poisson分布，总体均数𝜆 =总体方差σ²。

    ```stata
    poisson y x1 x2 x3 vce(robust) //泊松回归,robust是异方差情况
    poisson, irr //输出的是其均值变化倍数$exp(\beta)$，那么是期望发生次数𝜆的变化倍数
    margins x //边际处理，得出平均发生次数,其他值为均值
    estat gof //泊松分布是否符合我们的数据，需要拟合优度卡方检验在统计上不显著
    ```
    

4. **负二项回归**
    其服从的Poisson分布强度参数λ服从γ分布时，所得到的复合分布即为负二项分布
    在负二项分布中，λ 是一个随机变量，方差λ(1+kλ)远大于其平均数，k为非负值，表示计数资料的离散程度。当趋近于0时，则近似于Poisson分布，过离散是负二项分布相对于Poisson分布的重要区别和特点。
    可用拉格朗日算子统计量检验是否存在过离散，

    ```stata
    nbreg y x1 x2 x3, vce(robust) //负二项回归
    //负二项回归实际上和泊松回归一样，其数据过于离散，stata结果可以像泊松回归一样进行解释
    //同时会输出一个拉格朗日算子统计量检验是否存在过离散。若原假设成立就可以用
    ```

5. **零膨胀**
    其主要为了解决数据中存在大量的0值，同时其数据分布不符合泊松分布，因此需要进行零膨胀回归
    零膨胀模型有两部分，泊松计数模型和用于预测多余零的 logit 模型
    stata提供了Vuong统计量,Vuong”统计量很大 (为正数)，则应该选择零膨胀泊松回归
    ```stata
    zinb y x1 x2 x3, vce(robust) //零膨胀负二项回归
    //forcevuong: 用于比较 zinb和nb的模型效果
    //forcevuong不能与 vce() cluster standard error 同用, 可先比较两个模型后再聚合标准误
    zip y x1 x2 x3, vce(robust) //零膨胀泊松回归 参数与上同
    ```
    
6. **截尾回归**
    截尾回归是指因变量的观测值只能在某个区间内取值，而不能取到某个区间之外的值。截尾回归的模型是对数线性模型，其估计方法是最大似然估计法。
    ```stata
    truncreg y x1 x2 x3, ll(0) ul(1) //截尾回归 ll() 选项表示发生左截断的值，ul() 选项用于指示右截断值
    ```

7. **Tobit模型**  归并回归 (censored regression) 模型
    *当某个值大于或等于某一阈值时，就会出现上述归并，因此真实值可能等于某一阈值，但也可能更高*
    ```stata
    tobit y x1 x2 x3 //截尾回归 ll() 选项表示发生左截断的值，ul() 选项用于指示右截断值
    ```
8. **拟合优度**
    - Likelihood ratio index (LRI)似然比指数

        ```stata
        //需要储存模型
        estimates store 名称
        lrtest reduced_model full_model //需要其拒绝原假设
        ```

    - Akaike Information Criterion (AIC) 
        自动输出越小越好
    - Bayesian Information Criterion (BIC) 
        ```stata
        estat ic //输出AIC和BIC 选择最小的
        ```
    - Hit rate












## <div style="font-size:25px;">4.Matching</div>

1. **精确匹配**  
    ```stata
    //需要两个数据集
    merge 1:1 x using data2 //精确匹配,匹配后会生成一个新的数据集，其中包含了匹配成功的观测值
    ```

2. **模糊匹配**
    stata中没有模糊匹配的专有代码
    
    ```stata
    //同一数据集中两列中的数据
    matchit varname1 varname2 [, options]
    *- 两个不同数据集中的数据
    matchit idmaster txtmaster using "data2.dta"
    //quired(varlist) 为可选择的命令，其允许用户指定一个或多个必须完全匹配的变量
    reclink varlist using filename , idmaster(varname) idusing(varname) gen(newvarname) [required(varlist)]
    //method()：reclink支持多种匹配方法
    //idmaster(varname) idusing(varname)不一定相同
    ```

3. **倾向得分匹配PSM**  
    其具有降维的力量，同时避免了因协变量较多带来的维度诅咒问题。由于倾向得分匹配是被处理的概率，因此可以通过被处理概率来进行匹配。即可以用Logit或Probit模型来估计倾向得分
    这是由于倾向得分定理表示得分值也满足条件独立性，因此可以消除选择偏误。
    - 倾向得分匹配
    
        ```stata
        logit treat x1 x2 x3 //使用treat作为因变量，其他协变量进行估计得分，这估计的是协变量相同时被处理的概率
        predict pscore, pr
        psmatch2 treat, pscore(pscore) outcome(y) //进行匹配
        ```

    - 近邻匹配
    
        ```stata
        psmatch2 treat x1 x2, outcome(y) neighbor(n) //进行近邻匹配 1对n
        ```

    - 带卡尺近邻匹配
    
        ```stata
        psmatch2 treat x1 x2, outcome(y) caliper(0.1) n(1) //进行近邻匹配 1对1,卡尺为0.1，只有在卡尺内部才行
        ```

    - 核匹配
        核函数与其他的匹配不同，核函数会利用所有的数据，依据核函数进行加权。即对他们的Y进行加权
        ```stata
        psmatch2 treat x1 x2, outcome(y) kernel kerneltype(normal/biweight/epan/uniform/tricube) //进行核匹配
        ```

    
## <div style="font-size:25px;">5.Instrument Variable</div>

我们在使用工具变量时，需要进行检验，最常见的就是排除性和相关性。  
进行IV时我们需要讲故事，并且数据检验其合理性：同时其最基础的工具变量回归的代码如下

```stata
ivregress 2sls y (x1 = z1 z2) x2 x3, robust
```

### <div style="font-size:20px;">弱工具变量检验</div>

1. **F检验**
   
    ```stata
    reg y x ,robust  // OLS回归估计
    ivregress 2sls y (x=z1,z2),robust  // 2SLS回归估计   
    reg x z1 z2,robust  // 第一阶段回归估计
    test z1 z2   //查看是否有弱工具变量问题，F检验 大于10即可 F估计与弱IV的关系来自于causal inference
    ```

    <div style="color:blue;"><b>可以通过以上的第一阶段回归查看第一阶段的参数从而判断工具变量的相关性</b></div>  
    也可以比较OLS和2SLS的结果，看看是否有差异

2. **Cragg-Donald检验**  
   一般条件是同方差，无自相关

    ```stata
    ivreg2 y (x1 x2 = z1 z2), robust  //Cragg-Donald检验,要大于 10
    ```

3. **Kleibergen-Paap检验** 无iid假设

    ```stata
    ivreg2 y (x1 x2 = z1 z2), robust   //Kleibergen-Paap检验,要大于 10
    ```

### <div style="font-size:20px;">外生性（排除性）检验</div>

1. **Hausman检验**  

    ```stata
    //豪斯曼检验 这是在同方差条件下的检验
    reg y x1 x2
    estimates store ols
    ivregress 2sls y (x1 = z) x2
    estimates store iv
    hausman iv ols, constant sigmamore
    //chi - squared和p - value。p 小于0.05，拒原，认为变量是内生变量,p最好大一点
    ```

2. **DWH检验**  

    用上一个检验的结果就行，也会输出DWH检验的结果。这是在异方差条件下的检验

3. **GMM估计**
    
    ```stata
    ivregress gmm y (x1 = z1 z2), twostep robust     
    estat overid   //原假设：工具变量是有外生的
    ```

### <div style="font-size:20px;">过度识别检验</div>

1. **Sargan检验**  用于线性模型中的工具变量过度识别检验

    ```stata
    ivregress 2sls y (x1 = z1 z2)
    ```

2. **Anderson - Rubin 检验**  用于非线性模型或联立方程模型中的工具变量过度识别检验
    以联立方程模型为例

    ```stata
    sysreg (eq1: y1 = x1 x2 (y2 = z1 z2)) (eq2: y2 = x3 x4 (y1 = z3 z4))
    test [eq1_y2] [eq2_y1]  // 原假设是不存在过度识别问题
    ```

3. **Hansen J统计量** 非iid时用Hansen J统计量
   和Sargon检验类似 非iid时用Hassen统计量












































## <div style="font-size:25px;">实用小代码stata</div>

- ```stata
    count if contact == 1 //统计contact为1的个数
    ```

- ```stata
    drop if var==. //删除变量的缺失值
    ```

## <div style="font-size:25px;">一些方法</div>

- 证伪实验 ：证伪实验的目的不是证明某个假设是正确的，而是尝试找到证据来反驳它，证伪实验中，研究者会设计一个实验来检验假设的预测结果。如果实验结果与假设的预测不一致，那么就可以认为该假设被证伪了。例如：如果认为打电话对于02年的选举有影响，那证伪实验就是在98年进行打电话对于选举的影响，如果没有影响，那么就认为打电话对选举有影响（之前得出结论有影响）。