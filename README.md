Download Link: https://assignmentchef.com/product/solved-ece272a-homework-1
<br>
Welcome to the first homework assignment of the quarter!

Over the course of the quarter, we’re going to be helping you develop <em>tools </em>that apply machine learning and data science to real-world problems. This first assignment starts that off with teaching the initial phases of developing a tool: Exploring the data space of your problem and writing scripts to train models.

For this assignment, you work at <em>Insurance Corp Incorporated</em>, a company in the medical insurance space. You’ve just received an email from your boss:

From: <a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="fc9e938f8f919d92d29e938f8f85bc959f95929fd2919998">[email protected]</a>

To: <a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="8df4e2f8cde4eee4e3eea3e0e8e9">[email protected]</a>

Subject: New development direction

Hey, legal has just approved a new diabetes study for use in our tools. I’d like you to take a dig through it, try to make some classifiers to predict patient outcomes from the diagnostic data.

Since we’re only prototyping for now, you can just load and save the data in <strong>CSV</strong>, manipulate it in <strong>Numpy</strong>, plot in <strong>MatplotLib</strong>, and crib the classifiers from <strong>Sklearn</strong>.

I’ve attached the study data to this email. I’ve also pulled some files for patients who are due for tests before the due date I’ve set on this data exploration. See if you can predict how those tests are going to turn out early.

When you’ve got a handle on the modeling, get me a written report on which classifier model you think is our best bet. I hear legal’s getting into a fight with the government over some of our models not being ”explainable” so, try a <strong>DecisionTree </strong>and, if that’s not good enough, get back on why.

Hope to see you in the happy hour Zoom on friday!

<em>We are not planning a real Zoom happy hour; the above email is fictional.</em>

<h1>2           Dataset</h1>

We are using data from the <em>Pima Indians Diabetes Database</em>, a dataset of medical history metrics and diabetes outcomes – who got diabetes, who did not. We’re making use of a subset, narrowed to females of at least 21 years of age.

The file is provided in a format known as Comma Separated Values (CSV). You can open it as raw text to take a look! The data columns we have are Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, and Outcome. All of these fields come in numbers, but not all fields have data. Where the data is missing, the field has been left at 0.

Outcome is the one you want to predict! If it’s a 1, the patient had or developed diabetes during the period of the study. If it’s a 0, they did not.

To load our data, we’ll use Python’s built-in csv module. The shortest snippet to load the whole file is:

import csv with open(’diabetes.csv’,’r’) as file: reader = csv.reader(file) column_headers = next(reader) data_rows = list(reader)

You’re not required to use this snippet. If this is at all confusing, we discuss loading and preprocessing the data in more depth in the <em>Getting Started </em>guide.

<h1>3           Recommended Flow</h1>

Overall, we suggest that you

<ol>

 <li>Make a function to load your data and split it into samples and labels</li>

 <li>Make a function to plot your data, to see what it looks like.</li>

 <li>Make a function to alter/preprocess your data (if you think you can ”feature engineer” it!)</li>

 <li>Make a function to split the given data between a set of training data andset of validation data (See: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"><strong>train</strong></a> <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"><strong>test</strong></a> <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"><strong>split on SciKit Learn</strong></a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html">)</a></li>

 <li>Train a classifier (See: <a href="https://scikit-learn.org/stable/supervised_learning.html"><strong>Supervised Learning on SciKit Learn</strong></a><a href="https://scikit-learn.org/stable/supervised_learning.html">)</a></li>

 <li>Measure its performance (See: <a href="https://scikit-learn.org/stable/modules/cross_validation.html"><strong>Cross-Validation on SciKit Learn</strong></a><a href="https://scikit-learn.org/stable/modules/cross_validation.html">)</a></li>

 <li>Save plots/metrics of its performance</li>

 <li>Repeat the previous three steps until satisfied</li>

 <li>Predict the data labels in <em>csv </em>using a trained classifier</li>

</ol>

<h1>4           Problem Formulation</h1>

<table width="427">

 <tbody>

  <tr>

   <td width="7"> </td>

   <td colspan="3" width="167"><strong>Input</strong></td>

   <td width="58"><strong>Range</strong></td>

   <td width="203"><strong>Description</strong></td>

   <td width="7"> </td>

  </tr>

  <tr>

   <td width="7"> </td>

   <td colspan="3" width="167">Pregnancies</td>

   <td width="58">[0,∞)</td>

   <td width="203">Pregnancies the patient has had</td>

   <td width="7"> </td>

  </tr>

  <tr>

   <td width="7"> </td>

   <td colspan="3" width="167">Glucose</td>

   <td width="58">[0,∞)</td>

   <td width="203">Blood glucose level</td>

   <td width="7"> </td>

  </tr>

  <tr>

   <td width="7"> </td>

   <td colspan="3" width="167">BloodPressure</td>

   <td width="58">[0,∞)</td>

   <td width="203">Blood pressure</td>

   <td width="7"> </td>

  </tr>

  <tr>

   <td width="7"> </td>

   <td colspan="3" width="167">SkinThickness</td>

   <td width="58">[0,∞)</td>

   <td width="203">Thickness of the skin</td>

   <td width="7"> </td>

  </tr>

  <tr>

   <td width="7"> </td>

   <td colspan="3" width="167">Insulin</td>

   <td width="58">[0,∞)</td>

   <td width="203">Blood insulin level</td>

   <td width="7"> </td>

  </tr>

  <tr>

   <td width="7"> </td>

   <td colspan="3" width="167">BMI</td>

   <td width="58">[0,∞)</td>

   <td width="203">Body-mass index</td>

   <td width="7"> </td>

  </tr>

  <tr>

   <td width="7"> </td>

   <td colspan="3" width="167">DiabetesPedigreeFunction</td>

   <td width="58">[0,1)</td>

   <td width="203">Familial history of diabetes</td>

   <td width="7"> </td>

  </tr>

  <tr>

   <td width="7"> </td>

   <td colspan="3" width="167">Age</td>

   <td width="58">[0,∞)</td>

   <td width="203">Patient age</td>

   <td width="7"> </td>

  </tr>

  <tr>

   <td colspan="2" width="68"><strong>Output</strong></td>

   <td width="58"><strong>Range</strong></td>

   <td colspan="4" width="315"><strong>Description</strong></td>

  </tr>

  <tr>

   <td colspan="2" width="68">Outcome</td>

   <td width="58">0 or 1</td>

   <td colspan="4" width="315">Prediction for whether the patient will get diabetes</td>

  </tr>

  <tr>

   <td width="7"></td>

   <td width="61"></td>

   <td width="58"></td>

   <td width="47"></td>

   <td width="58"></td>

   <td width="190"></td>

   <td width="6"></td>

  </tr>

 </tbody>

</table>

Your task is to create a classifier that converts diagnostic and historic data about patients into a prediction for whether or not they will develop diabetes.

You are required to use the Python scikit-learn library to construct your models. You are required to use the <strong>DecisionTree </strong>classifier and at least two of the following others:

<ul>

 <li><strong>Linear Discriminant Analysis </strong>(LDA)</li>

 <li><strong>Na¨ıve Bayes</strong></li>

 <li><strong>Nearest Neighbors</strong></li>

 <li><strong>Support Vector Machine </strong>(SVM)</li>

</ul>

Not all of these methods can achieve good results! Links to the documentation for each of these classifiers is available on <a href="https://scikit-learn.org/stable/supervised_learning.html"><strong>Supervised Learning in SciKit </strong></a><a href="https://scikit-learn.org/stable/supervised_learning.html"><strong>Learn</strong></a><a href="https://scikit-learn.org/stable/supervised_learning.html">)</a>

After you have trained your algorithms and selected the one you think is best, train it on the whole training set, then predict on the data in <em>unknowns.csv</em>. Make a new CSV file, <em>score.csv</em>, with only one column, the predicted outcomes, and submit it to the autograded dropbox. Submit your report to the nonautograded dropbox.

You also must write a brief report answering the following questions:

<ul>

 <li>Which algorithm did you decide was best?</li>

 <li>Describe in your own words how each algorithm you used classifies patients.</li>

 <li>Some models require setting ”hyperparameters” (such as the SVM tolerance and kernel function, or Nearest Neighbors’ number of neighbors checked.) Which hyperparameters did you have to tune? How did you decide on their values? <strong>Show at least one plot of a classifier’s performance versus one of its hyperparameters.</strong></li>

 <li>When choosing your final model, what percentage split did you give beteween training and validation data? Why did you make that choice? <strong>Show at least one scatterplot marking mispredicted datapoints.</strong></li>

 <li>Show a diagram of your DecisionTree classifier’s decision function. Does this decision function provide any hints about risk factors for diabetes?</li>

 <li><strong>Show all tested classifiers’ results using </strong><em>confusion matrices </em>over your validation set. Which models overfit to the data? Underfit? Which had the best accuracy?</li>

</ul>

The report need not be excessively long. If you’re spending more time putting together the report than you spent playing with the algorithms and data, feel free to drop by TA office hours for clarification on what we’re looking for!