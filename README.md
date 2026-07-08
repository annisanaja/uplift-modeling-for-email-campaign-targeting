# Uplift Modeling for Email Campaign Targeting

Causal inference project using the Hillstrom email marketing dataset to determine which customers convert *because of* an email campaign, rather than which customers simply convert. Standard classification models cannot make this distinction, since they cannot tell a customer who would have purchased anyway apart from one who was genuinely persuaded by the email. This project builds and evaluates uplift models to answer that question directly, then translates the results into a targeting recommendation.

## Business Question

Should the company continue its email campaigns, and can targeting be made more precise to reduce send volume while preserving impact?

## Data Source

[Kevin Hillstrom MineThatData E-Mail Analytics](https://www.kaggle.com/datasets/bofulee/kevin-hillstrom-minethatdata-e-mailanalytics), donated by BOFU-LEE on Kaggle. Originally published by Kevin Hillstrom on the MineThatData blog as part of a public email analytics challenge.

64,000 customers who purchased within the last twelve months were randomly split into three groups: a men's product email, a women's product email, or no email. Outcomes (site visit, conversion, spend) were tracked for two weeks afterward. Because assignment was random, this is a genuine randomized controlled experiment, which makes it a clean setting for causal inference.

## Method

1. Split each campaign (Women's Email vs. No Email, Men's Email vs. No Email) into a proper train/test split, stratified on treatment assignment
2. Built a baseline Random Forest classifier predicting conversion directly, to illustrate why a standard classifier cannot separate incremental effect from baseline behavior
3. Built a T-Learner uplift model (two Random Forest models, one trained on treated customers and one on control customers) for each campaign
4. Evaluated targeting reliability using Qini AUC on a held-out test set, since standard accuracy metrics cannot capture incremental causal effect
5. Confirmed the aggregate campaign effect using a chi-square test on the full dataset, since the test split alone does not carry enough statistical power given how rare conversions are
6. Checked whether new versus returning customers respond differently within each campaign, using the same held-out test set throughout to avoid validating a model against data it was already trained on
7. Refit a final model on the complete dataset to generate a deployable targeting list, kept separate from the evaluation step above

## Key Findings

| Metric | Women's Email | Men's Email |
|---|---|---|
| Control conversion rate | 0.57% | 0.57% |
| Treatment conversion rate | 0.88% | 1.25% |
| Relative lift | +54% | +118% |
| Statistical significance | p = 0.0002 | p < 0.0001 |
| Qini AUC (held-out test set) | +0.076 | −0.025 |

Both campaigns produce a real, statistically significant lift in conversion. The Men's campaign has the larger effect, roughly double the relative lift of the Women's campaign.

The two campaigns diverge sharply on whether that effect can be targeted at specific customers. The Women's model shows a small but genuine positive Qini AUC, meaning it ranks customers by responsiveness a bit better than random. The Men's model shows a negative Qini AUC, meaning its ranking performs worse than picking customers at random, so attempting to target it would likely reduce results rather than improve them.

## Recommendation

Both campaigns should continue, since each has a proven positive effect. The Men's campaign should be prioritized for budget given its stronger return, and should be sent to the full eligible list rather than filtered, since there is no reliable pattern showing who benefits more within that group. The Women's campaign can be sent more selectively, targeting the subset of returning customers with a positive predicted uplift score, since this is the group with the clearest supporting evidence. New customers, across both campaigns, should continue receiving their respective email without filtering, since the available features (built mostly from purchase history) do not yet support singling anyone out within that group.

The clearest next step for improving targeting further is adding features around actual email engagement, such as past opens or clicks, rather than relying on purchase history alone. That kind of signal tends to carry much more predictive weight for who responds to a new campaign, and could be what moves the Men's campaign from unreliable to usable for targeting.

## Repository Structure
```
uplift-email-targeting/
├── README.md
├── requirements.txt
├── [naja]_uplift_modeling_targetedmails.py
├── cust-uplift.csv
└── outputs/
    ├── qini_curve_womens.png
    ├── qini_curve_mens.png
    ├── recommended_targets_womens_returning.csv
    └── recommended_targets_mens_returning.csv
```
## Running This Project

```bash
pip install -r requirements.txt
python uplift_modeling.py
```

Place `cust-uplift.csv` (the Hillstrom dataset) in the project root before running.

## Tools

Python, pandas, scikit-learn, scikit-uplift, scipy, matplotlib

## A Note on Methodology

An earlier version of this analysis validated model quality using the same data the final model was trained on, which produced misleadingly strong results. The version in this repo keeps model evaluation (on a genuine held-out test set) and final targeting list generation (refit on all available data, standard practice before deployment) as two separate steps, so the reported Qini AUC scores reflect real performance on unseen customers rather than the model recalling its own training data.
