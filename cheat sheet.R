
# array start with 1 not 0

setwd("C:/Users/chirag/Desktop/Data Science/hw/hw4")
x<-1:5

typeof(x)
class(x)
x1<-c(1,2,3,4)
typeof(x1)
class(x1)


xp=c(rep(1,100))


a=2
print(cat(a,"Hi"))

fun1<-function(c){
  print("hello")
  c1=1
  print(c)}

fun1("chirag")


M = matrix( c('a','a','b','c','b','a'), nrow = 2, ncol = 3, byrow = TRUE)
M = matrix( c('a','a','b','c','b','a'), 2,3)



a <- array(c(1,2,3,4,5,6,7,8,9,10,11,12),dim = c(3,3,2))
print(a)


# Create a vector.
apple_colors <- c('green','green','yellow','red','red','red','green')

# Create a factor object.
factor_apple <- factor(apple_colors)

# Print the factor.
print(factor_apple)
levels(factor_apple)
print(nlevels(factor_apple))





x<-1:5
for (a in x){
  print(1+a)
  print("heloo",a)}


x<-"hello how are you"
nchar(x)
toupper(x)
tolower(x)
substring(x,2,4)
spl<-strsplit(x,c(" ","o"), fixed = FALSE, perl = FALSE, useBytes = FALSE)


y="hi"
print(paste(x,y))



city <- c("Tampa","Seattle","Hartford","Denver")
state <- c("FL","WA","CT","CO")
zipcode <- c(33602,98104,06161,80294)

# Combine above three vectors into one data frame.

addresses <- cbind(city,state,zipcode)

new_var<-ifelse(state=="FL",1,0)
new_df<-data.frame(addresses,new_var)
names(new_df)
str(new_df)
new_df$
removed_fist<-new_df[,-1] # removes first columns  
new_df[1] # select first column
new_df[c(1,3),] # select all 1,3 rows all columns
table(new_df$new_var) # gives vlookup count


d1<-data.frame(addresses)  # matrix to data frame
dim(d1)

m2<-as.matrix(d1) # dataframe to matrix

# rbind fuction appends row by row
# cbinnd function appends columns by columns






