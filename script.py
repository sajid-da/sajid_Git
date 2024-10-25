
pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scmGit(branches: [[name: 'main']], extensions: [], userRemoteConfigs: [[credentialsId: 'sajid-da', url: 'https://github.com/sajid-da/sajid_Git.git']])
            }
        }
        stage('Build') {
            steps {
                git branch: 'main', credentialsId: 'sajid-da', url: 'https://github.com/sajid-da/sajid_Git.git'
                bat 'python sout.py'
                
            }
        }
        stage('Test') {
            steps {
                echo "Testing is done"
            }
        }
    }
}
