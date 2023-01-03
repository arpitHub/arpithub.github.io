---
layout: post
title: Odds and Odds Ratio
subtitle: Explaining Odds and Odds Ratio with the help of example
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
gh-repo: arpithub/arpithub.github.io
gh-badge: [star, fork, follow]
tags: [statistics,book]
comments: true
---

Currently, I am reading the book **The Art of Statistics** by Sir David John Spiegelhalter and wanted to touch on the Odds and Odds ratio.
We see these concepts everywhere, odds of an event, odds of winning something.
The definition of Odds is:
 __Chance of an event happening / Chance of an event not happening__

Odds Ratio: It's the ratio of odds. 
__Odds of an event happening / Odds of an event not happening__

In the medical domain, the Odds ratio helps to understand if an intervention works or not and to what degree.

For example, we want to test the effectiveness of a vaccine campaign. We will first calculate the odds of the Intervention group and the odds of the Control group.
Let's say there are 500 people in the **Intervention group** which consists of people seen the vaccine campaign and the **Control group** which consists of 500 people who haven't seen the campaign.
Among the Intervention group, 400 people have taken the vaccine, and the remaining 100 people haven't taken the vaccine.
In the Control group, 275 people have taken the vaccine and the remaining 225 people haven't taken the vaccine.

Odds of getting the vaccine in the Intervention group = No. of people who took the vaccine / No. of people who haven't taken the vaccine

Plugging in the numbers from the above example,
Odds of getting vaccine in Intervention group = 400/100 = 4

Odds of getting the vaccine in the Control group = No. of people who took the vaccine / No. of people who haven't taken the vaccine

Plugging in the numbers from the above example,
Odds of getting vaccine in Intervention group = 275/200 ~ 1.23

To find the effectiveness of the vaccine campaign we will calculate the Odds Ratio.
ie. Ratio of Odds of Intervention group (saw campaign) to Odds of Control group (didn't see campaign).

Plugging in the numbers:
Odds Ratio = 4/1.23 = 3.25

How to interpret Odds Ratio:
If, 
Odd Ratio = 1 then it means there wasn't any effect of an action on a task, here campaign (Action) has no effect on taking the vaccine (task).

Odd Ratio > 1 means people who were exposed to action were more likely to perform a task, here people who saw the campaign were more likely to take the vaccine as compared to people who haven't seen the campaign.

Odd Ratio < 1 means people who were exposed to action were less likely to perform a task, here people who saw the campaign were less likely to take the vaccine as compared to people who haven't seen the campaign.

In our example since the Odds Ratio is greater than 1 that means the vaccine campaign was effective as people who saw the campaign (Intervention group) are more likely to take the vaccine.
The Odds ratio value of 3.25 tells us that people in the Intervention group have 3 times more odds of taking the vaccine as compared to the Control group.

Note - Odds Ratio greater or less than 1 can be a positive or negative finding depending on the outcome.
The Odds Ratio is like R-squared which shows the relationship between two things. In the above example, a relationship between Exposing to Vaccine Campaign and Getting the vaccine.

To further validate the relationship between the vaccine campaign and getting the vaccine, we can perform some Significant tests. We can check if the Odds Ratio is statistically significant or not.
Here are the 3 Signifincant Tests we can perform:
1. Fisher's Exact Test
2. Chi-Square Test
3. Wald Test

Although the Odds Ratio is a powerful tool it should be used very carefully.

References:
1. [https://www.youtube.com/watch?v=5zPSD_e_N04]
2. [https://www.youtube.com/watch?v=8nm0G-1uJzA]
