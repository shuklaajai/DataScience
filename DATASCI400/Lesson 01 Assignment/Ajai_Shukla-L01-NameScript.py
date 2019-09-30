"""
# UW Data Science
# To run the code, hilight lines by selecting the line number
# and then press CTRL+ENTER or choose Run Selection from the toolbar.
"""
"""
# Function, named "my_name" that does not require arguments and returns your name as a string.
#def my_name():

  print("Ajai Shukla")
 
my_name()

#####################  
#Call to "my_name" where the output is printed to the console with a print statement.
#def my_name(name="Ajai Shukla"):
   
    print(name)
    
my_name()

###################################################################################
#An import statement for package datetime.datetime that uses the "as" keyword

import datetime
import datetime as dt 
dt is datetime 

#################################################################################
#A function called “date_and_time” that uses the datetime package and returns a string with the current date and time
import datetime
#def date_and_time(now="current_time"):
    now = datetime.datetime.now()
    print ("Current date and time : ")
    print (now.strftime("%m-%d-%Y %H:%M:%S"))
    
date_and_time()

 ###################################################################################
 #Call to "date_and_time" where the output is printed to the console with a print statement.
 
#def date_and_time(now = datetime.datetime.now()):
     #now = datetime.datetime.now()
     print("Current date and time:", now )
     
date_and_time()

    
