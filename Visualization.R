library(ggplot2)
library(dplyr)
library(srvyr)
fake=read.csv(file="fake_job_postings.csv")

# Data Exploration
unique_columns_count <-  fake %>% 
  summarise(n_title = n_distinct(title),
            n_location = n_distinct(location),
            n_department = n_distinct(department),
            n_salary_range = n_distinct(salary_range),
            n_employment_type = n_distinct(employment_type),
            n_required_experience = n_distinct(required_experience),
            n_required_education = n_distinct(required_education),
            n_industry = n_distinct(industry),
            n_function = n_distinct(function.),
            n_fraudulent = n_distinct(fraudulent))
print(unique_columns_count)

# Distribution of jobs
job_distribution = fake %>% group_by(fraudulent) %>%  ggplot(aes(fraudulent, group = fraudulent)) + 
  geom_bar(aes(fill = fraudulent), stat = "count") + 
  theme(axis.text.x = element_text(angle = 90, size = 10)) + 
  geom_text(aes(label=..count..),stat='count',position=position_stack(vjust=0.5)) + 
  ggtitle("Genuine vs. Fraud Jobs") + xlab("Fraud Flag") + ylab("Job Count") + theme_bw()
job_distribution

# Distribution of degrees
degree_distribution = fake %>% group_by(required_education, fraudulent) %>% summarise(count = n())
degreedistribution = degree_distribution %>%  ggplot(aes(reorder(
  degree_distribution$required_education, -degree_distribution$count), degree_distribution$count)) +
  geom_bar(stat = "identity", aes(fill = fraudulent)) + 
  theme(axis.text.x = element_text(angle = 90, size = 10)) + 
  ggtitle("Jobs Per Required Education Feature") + xlab("Required Education") + ylab("Job Count")
degreedistribution 

# Distribution of experience
experience_distribution = fake %>% group_by(required_experience, fraudulent) %>% summarise(count = n())
experiencedistribution=experience_distribution %>%  ggplot(aes(reorder(
  experience_distribution$required_experience, -experience_distribution$count), experience_distribution$count)) +
  geom_bar(stat = "identity", aes(fill = fraudulent)) + 
  theme(axis.text.x = element_text(angle = 90, size = 10)) + 
  ggtitle("Jobs Per Required Experience Feature") + xlab("Required Experience") + ylab("Job Count")
experiencedistribution

# Distribution of Employment Types
employment_type_distribution <- fake %>% group_by(employment_type, fraudulent) %>% summarise(count = n())
employmenttype_distribution=employment_type_distribution %>%  ggplot(aes(reorder(
  employment_type_distribution$employment_type, -employment_type_distribution$count), employment_type_distribution$count)) +
  geom_bar(stat = "identity", aes(fill = fraudulent)) + 
  theme(axis.text.x = element_text(angle = 90, size = 10)) + 
  ggtitle("Jobs Per Required Employment Types Feature") + xlab("Employment Type") + ylab("Job Count")
employmenttype_distribution

# Distribution of experience and education
fake %>% group_by(required_education) %>% ggplot(aes(x = required_education), group = required_experience) +
  geom_bar(aes(fill = fake$required_experience), stat = "count") + 
  theme(axis.text.x = element_text(angle = 90, size = 10)) + 
  ggtitle("Jobs Per Required Education and Experience") + xlab("Required Education") + 
  ylab("Job Count") + labs(fill='Required Experience')

# Distribution of experience and employment type
fake %>% group_by(employment_type) %>% ggplot(aes(x = employment_type), group = required_experience) +
  geom_bar(aes(fill = fake$required_experience), stat = "count") + 
  theme(axis.text.x = element_text(angle = 90, size = 10)) + 
  ggtitle("Jobs Per Required Experience") + xlab("Employment Type") + 
  ylab("Job Count") + labs(fill='Required Experience')

# Distribution of education and employment type
fake %>% group_by(employment_type) %>% ggplot(aes(x = employment_type), group = required_education) +
  geom_bar(aes(fill = fake$required_education), stat = "count") + 
  theme(axis.text.x = element_text(angle = 90, size = 10)) + 
  ggtitle("Jobs Per Required Education") + xlab("Employment Type") + 
  ylab("Job Count") + labs(fill='Education Level')












