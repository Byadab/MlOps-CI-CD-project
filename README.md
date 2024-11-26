## End to end machine learnign project# MlOps-CI-CD-project
###### MlOps-CI-CD-project
## AWS Deployment


### Steps:
1- Docker builder check

2- Github workflow setup

3- AWS - IAM user creation

4- AWS - ECR setup

5- AWS - EC2 setup --> create virtual cloud machine 

6- Cretaed github action runner in EC2 using all the provided commands in setings->actions->runners->new runner
    
    - Add secrets and variables in github repo settings->secrets and variables->actions
        
        - AWS_ACCESS_KEY_ID
        
        - AWS_SECRET_ACCESS_KEY
        
        - AWS_REGION
        
        - AWS_ECR_LOGIN_URI
        
        - ECR_REPOSITORY_NAME

### Docker Setup In EC2 commands to be Executed
##### optinal

sudo apt-get update -y

sudo apt-get upgrade

##### required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu 

newgrp docker -- create a group named docker

