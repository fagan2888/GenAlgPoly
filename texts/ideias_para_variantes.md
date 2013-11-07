#Ideias para variantes e melhorias

1. cross-validation (lambda, elitismo, etc)
2. fazer classificação
2. GA ensemble
    3. FC: o ensemble é a elite de uma população => mas é preciso atender à variedade genética;
    0. JPN: o ensemble é um conjunto de polinómios, cada um obtido a partir de aplicações distintas da versão _single_ do algoritmo;
3. regressão/classificação com funções polinomiais implícitas
5. reter parte da população e posteriormente juntar novas observações para reduzir a computação do GA em aplicações onde
    6. a velocidade é importante e
    7. "chegam" novas observações frequentemente
4. usar dados do kaggle

## Tarefas para Revisão

> Com o feedback da primeira submissão, temos matéria para melhorar a versão actual e resubmeter noutro journal.

1. Relacionar o nosso trabalho com "_previous approaches using GA to solve polynomial regressions_";
2. Explicitar os benefícios do nosso método, melhorando a redação das comparações ("_the proposed method showed only "competitive" result compared to other approaches_");
3. Atender à hipótese de saber se "_this idea could be improved with different parameter tuning methods instead of genetic algorithm_";
4. Melhorar a escrita atendendo a que "_There are more than necessary use of colloquial style writings_".
5. <span style ="color: crimson">Ver trabalho prévio em _Advances in data-driven analyses and modelling using EPR-MOGA_.</span>


### Comentários à 1ª versão
> AE: Thank you for submitting your manuscript "Selection of Polynomial Features by Genetic Algorithms" for publication in Pattern Recognition Letters.  The submission has been reviewed by two expert referees.  My unpleasant task as an Associate Editor of the journal is to inform you that the manuscript has been found in its current form inappropriate for publication and I need to recommended to reject it. The reviewers' opinion was that the proposed method  did not have enough novelty and the experiments with it were not sufficiently convincing to prove the benefits it could provide.  You will find the detailed comments of Reviewer #2 below. 

#### Reviewer #1's opinion in short,
> communicated via the confidential section to the editor, was that **"this idea could be improved with different parameter tuning methods instead of genetic algorithm"**.  I hope these comments will be useful for you.  After substantial improvements, you may consider resubmitting your work again as a new
manuscript for review.
Thank you again for considering Pattern Recognition Letters as a forum for your publication.

#### Reviewer #2: 
> The authors proposed a method that utilizes Genetic Algorithm and Linear Regression to solve polynomial regression problems. The proposed method was compared with other approaches to solve polynomial regressions. Overall, **it is not clear what benefit can be achieved from the proposed method compared to previous approaches**. Moreover, it is hard to recognize the difference (or novelty) of the proposed method compared to **previous approaches using GA to solve polynomial regressions**. Novel benefits of the proposed method need to be specifically described compared to previous approaches using GA for polynomial regressions, and it should be supported by experimental comparisons. For summary, in this reviewer's opinion, the followings can be revised to improve the manuscript:
> 
> * Major comments
>
> 1. The most significant issues in this manuscript are two-folds. First, **the proposed method showed only "competitive" result compared to other approaches**. Second, **there is no comparison with previous approaches** using GA for polynomial regressions. For these reasons, it is not easy to find a reason to use the proposed method, and it is not clear what benefit the proposed method has  compared to previous methods using GA for polynomial regressions. This manuscript will have contributions to research community only when such points are revised properly.
> 
> Authors claim that (in line 223 - 224) **using GA for polynomial regressions is a meaningful approach. However, that point must have been shown already by other previous approaches using GA for polynomial regressions**. Without additional (and more necessary) comparisons and proven benefits of the proposed method, it is hard to recognize a meaningful contribution from this manuscript.
>
> 2. In lines 130 - 131, authors **claimed that considering more complex mappings would negatively impact the performance. Providing specific reasons needs to follow**.
>
> * Minor comments
>
>1. There are more than necessary **use of colloquial style writings**.
