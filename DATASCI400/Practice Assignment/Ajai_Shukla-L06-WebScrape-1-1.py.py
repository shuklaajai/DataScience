"""
# UW Data Science
Create a new Python script that will run in the grader's environment using only spyder's run file button. 
Include the following:

1.Import statements for necessary package(s)
2.Read in an html page from a freely and easily available source on the internet. The html page must contain at least 3 links.
3.Write code to tally the number of links in the html page.
4. Use print to present the tally
"""
#########################################################
# Import statements for necessary package(s)
###########################################################
import requests
from bs4 import BeautifulSoup 


############################################################
# Read in an html page from a freely and easily available source on the internet. The html page must contain at least 3 links.
##########################################################################################################################
url = "https://wiki.python.org/moin/IntroductoryBooks"
 
# now we can use urllib to pull down the html and store it as a request object
response = requests.get(url)

# here can view the page headers
print(response.headers)

# Grab the page content
content = response.content
print(content)
# to notify beautifulsoup about what type of HTML format we are working with
soup = BeautifulSoup(content, "lxml")

# now that we've pulled down the page content, let's use beautifulsoup to 
# convert it to something that we can read, here we add the lxml tag 
# to notify beautifulsoup about what type of HTML format we are working with
soup = BeautifulSoup(content, "lxml")

#####################################################################################################################
## Use print to present the tally
##################################################################################################################
# we can use BeautifulSoup's prettify function to print the html from our soup object in a more readable format 
# so that we can figure out what to grab out
print(soup.prettify())
# to get the information insided title tags we could do: 
print(soup.title)

# We can even convert this to a string for downstream use
print(soup.title.string)
# let's say we want to grab all info inside a tags, which commonly contains links 
all_a = soup.find_all("a")

# this returns in iterable object that we can loop through
for x in all_a:
    print(x)
#  we can look inside "a" tags for the "http" tags: 
all_a_https = soup.find_all("a", "https")   

for x in all_a_https:
    print(x)
    # we can access items inside the iterable just like with a regular python list
print(all_a_https[0])

# and we can even convert that result diretly to a string
print(all_a_https[0].string)
# that are nested inside of the dev tag that we pulled out: 

for x in all_a_https:
    print(x.attrs['href'])
    
# Now that we know how to pull out data, we can pull out the elements we need
# and automatically convert them into useful python data stucutres, like a dictionary: 
    
data = {}    
for a in all_a_https: 
    title = a.string.strip() 
    data[title] = a.attrs['href']

print(data)
############################################################################################################################