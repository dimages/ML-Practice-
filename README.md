###Флоу запуска проекта

##Скачиваем и запускаем виртуальное окружение:

pip install virtualenv

virtualenv myenv

myenv\Scripts\Activate

##Устанавливаем библиотеки 
pip install -r requirements.txt

##Запускаем проект
uvicorn main:app --reload
