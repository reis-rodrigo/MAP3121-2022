"""
Tarefa 2 - Fórmulas de Integração Numérica de Gauss - MAP3121

    (Data de entrega: 05/06/2022)

    Rodrigo Gebara Reis - NUSP: 11819880
    Victor Rocha da Silva - NUSP: 11223782
"""

# Os testes criados entre as funções são apenas para validação (não foram exigi).

import numpy as np
import time


def nos_pesos(n): # recupera nós e pesos a partir do arquivo .txt fornecido

  assert (n == 6 or n == 8 or n == 10), "O valor utilizado para os nós e pesos não é válido. Tente usar 6, 8 ou 10."

  if n == 6:
    x = np.array([-0.2386191860831969086305017, -0.6612093864662645136613996, -0.9324695142031520278123016, 0.2386191860831969086305017, 0.6612093864662645136613996, 0.9324695142031520278123016])
    w = np.array([0.4679139345726910473898703, 0.3607615730481386075698335, 0.1713244923791703450402961, 0.4679139345726910473898703, 0.3607615730481386075698335, 0.1713244923791703450402961])

  elif n == 8:
    x = np.array([-0.1834346424956498049394761, -0.5255324099163289858177390, -0.7966664774136267395915539, -0.9602898564975362316835609,
                  0.1834346424956498049394761, 0.5255324099163289858177390, 0.7966664774136267395915539, 0.9602898564975362316835609])
    w = np.array([0.3626837833783619829651504, 0.3137066458778872873379622, 0.2223810344533744705443560, 0.1012285362903762591525314,
                  0.3626837833783619829651504, 0.3137066458778872873379622, 0.2223810344533744705443560, 0.1012285362903762591525314])

  elif n == 10:
    x = np.array([-0.1488743389816312108848260, -0.4333953941292471907992659, -0.6794095682990244062343274, -0.8650633666889845107320967, -0.9739065285171717200779640,
                  0.1488743389816312108848260, 0.4333953941292471907992659, 0.6794095682990244062343274, 0.8650633666889845107320967, 0.9739065285171717200779640])
    w = np.array([0.2955242247147528701738930, 0.2692667193099963550912269, 0.2190863625159820439955349, 0.1494513491505805931457763, 0.0666713443086881375935688,
                  0.2955242247147528701738930, 0.2692667193099963550912269, 0.2190863625159820439955349, 0.1494513491505805931457763, 0.0666713443086881375935688])

  return x, w


def integral(f, n): 
# Entradas: função f(x) e número de nós n
# Saída: resultado da integral de -1 a 1 de f(x)

  inicio = time.time()
  x, w = nos_pesos(n) # Recupera os valores fornecidos no arquivo .txt
  sum = 0
  
  for i in range(n): # Realiza a soma descrita no algoritmo deduzido (soma de w_j*f(x_j))
    sum += w[i]*f(x[i])

  print("Tempo:", time.time()-inicio)

  return sum # Retorna o valor final da soma, aproximadamente a integral definida


# Teste 1 [f(x)=np.exp(-x)+4*x]

#f = lambda x: np.exp(-x)+4*x
#calc = 2.350402387287602913764 # Resposta obtida utilizando calculadora
#result = integral(f, 6) # Resposta obtida a partir da função criada acima
#print('Resultado:', result, '\nErro:', abs(calc-result))


def integral_a_to_b(f, a, b, n):
# Entradas: função f(x), extremos a, b do intervalo de integração, número de nós n
# Saída: resultado da integral de a até b de f(x)
# Caso mais geral da função integral(f, n), aplica a fórmula de mudança de variável deduzida acima

  inicio = time.time()
  l = (b-a)/2 # Define-se dois parâmetros para a transformação linear: metade do tamanho do intervalo
  m = (a+b)/2 # e o ponto médio, segundo as fórmulas deduzidas

  y, w = nos_pesos(n) # Recupera os valores fornecidos no arquivo .txt

  x = l*y + m # Aplica a transformação linear do intervalo [-1,1] para [a,b]
  sum = 0
   
  for i in range(n): # Realiza a soma descrita no algoritmo deduzido (soma de w_j*f(x_j))
    sum += w[i]*f(x[i])

  sum = sum*l # Finaliza a aplicação da fórmula geral para [a,b] multiplicando por (b-a)/2

  print("Tempo:", time.time()-inicio)

  return sum # Retorna o valor final da soma, aproximadamente a integral definida


# Teste 2 [f(x)=np.exp(-x)+4*x, [-3,7]] 

#f = lambda x: np.exp(-x)+4*x
#calc = 100.08462504122211322472 # Resposta obtida utilizando calculadora
#result = integral_a_to_b(f, -3, 7, 6) # Resposta obtida a partir da função criada acima
#print('Resultado:', result, '\nErro:', abs(calc-result))


def integral_dupla(f, a, b, c, d, n):
# Entradas: função f(x,y), extremos a, b do intervalo de extremos fixos, c e d extremos variáveis (funções de uma variável), número de nós n
# Saída: resultado da integral dupla da região a <= x (ou y) <= b, c <= y (ou x) <= d 
# Utiliza a mesma estrutura do algoritmo de integral definida simples, com um loop interno ao já presente anteriormente.

  inicio = time.time()

  lx = (b-a)/2 # Define o tamanho de metade do intervalo fixo para transformação linear
  mx = (a+b)/2 # Define o ponto médio do intervalo fixo para transformação linear

  # Recupera os valores fornecidos no arquivo .txt. 
  #Note que pode-se lê-los apenas uma vez, alterando as transformações para cada y_j
  r, w = nos_pesos(n)
  x = lx*r + mx # Aplica a transformação linear à variável de extremos fixos

  sum = 0

  for i in range(n): # Para cada nó da variável de extremos fixos, aplica o algoritmo integral_a_to_b descrito acima
    ly = (d(x[i])-c(x[i]))/2 # Calcula metade do tamanho do intervalo
    my = (c(x[i])+d(x[i]))/2 # Calcula o ponto médio do intervalo

    y = ly*r + my # Aplica a transformação linear [-1,1]->[c,d]
    sum_y = 0
    
    for j in range(n):
      sum_y += w[j]*f(x[i], y[j]) # Realiza a soma descrita no algoritmo deduzido (soma de w_j*f(x_i,y_j))

    sum += w[i]*sum_y*ly # Finaliza a aplicação da fórmula geral para [c,d] multiplicando por (d-c)/2

  sum = sum*lx # Finaliza a aplicação da fórmula geral para [a,b] multiplicando por (b-a)/2

  print("n = ", n, "\nTempo: ", time.time()-inicio, "\nResultado da integral: ", sum)

  return sum # Retorna o valor final da soma, aproximadamente a integral dupla definida


# Teste3 3 [f(x,y)=(4*y)/(x**3+2), [1,3], [0,2x]] 

#f = lambda x, y: (4*y)/(x**3+2)
#c = lambda x : 0
#d = lambda x : 2*x
#calc = 6.049822776848971562101 # Resposta obtida utilizando calculadora
#result = integral_dupla(f, 1, 3, c, d, 6) # Resposta obtida a partir da função criada acima
#print('\nErro:', abs(calc-result))


## TAREFA

# Exemplo 1

# (a)
# Para o volume do cubo, definimos a função constante 1:
f = lambda x, y : 1

# Definimos os extremos fixos:
a = 0
b = 1

# O algoritmo precisa de dois extremos que variam com x ou y, então definimos também:
c = lambda x : 0
d = lambda x : 1

n = [6, 8, 10]
for i in n:
  r = integral_dupla(f, a, b, c, d, i)
  print('Erro:', abs(r-1), '\n')
  
  
# (b)
# Para o volume da pirâmide, definimos a função f(x,y) = 1 - x - y (deduzido acima):
f = lambda x, y : 1 - x - y

# Definimos os extremos fixos:
a = 0
b = 1

# O algoritmo precisa de dois extremos que variam com x ou y, então definimos também:
c = lambda x : 0
d = lambda x : -x + 1

n = [6, 8, 10]
for i in n:
  r = integral_dupla(f, a, b, c, d, i)
  print('Erro:', abs(r-(1/6)), '\n')
    

# Exemplo 2

# A função a ser integrada é a mesma para as duas ordens:
f = lambda x, y : 1

# Definimos os extremos fixos:
a = 0
b = 1

# Extremos variáveis da integral da esquerda:
c1 = lambda x : 0
d1 = lambda x : 1-x*x

# Extremos variáveis da integral da direita:
c2 = lambda x : 0
d2 = lambda y : np.sqrt(1-y)


# (Integral da esquerda)

n = [6, 8, 10]
for i in n:
  r = integral_dupla(f, a, b, c1, d1, i)
  print('Erro:', abs(r-(2/3)), '\n')


# (Integral da direita)

for i in n:
  r = integral_dupla(f, a, b, c2, d2, i)
  print('Erro:', abs(r-(2/3)), '\n') 
  
  
# Exemplo 3

# (Area)

# A função a ser integrada é a mesma para as duas ordens:
f = lambda x, y : (((y**2/x**4)+(1/x**2))*np.exp((2*y)/(x))+1)**0.5

# Definimos os extremos fixos:
a = 0.1
b = 0.5

# Extremos variáveis da integral da esquerda:
c = lambda x : x**3
d = lambda x : x**2

# O resultado utilizado para calcular o erro foi obtido por meio de calculadora
for i in n:
  r = integral_dupla(f, a, b, c, d, i)
  print('Erro:', abs(r-0.105498), '\n')
  
# (Volume)

# A função a ser integrada é a mesma para as duas ordens:
f = lambda x, y : np.exp(y/x)

# Definimos os extremos fixos:
a = 0.1
b = 0.5

# O resultado utilizado para calcular o erro foi obtido por meio de calculadora
for i in n:
  r = integral_dupla(f, a, b, c, d, i)
  print('Erro:', abs(r-0.0333056), '\n')
# Extremos variáveis da integral da esquerda:
c = lambda x : x**3
d = lambda x : x**2


# Exemplo 4

# (a)

# A função a ser integrada é a mesma para as duas ordens:
f = lambda x, y : y

# Definimos os extremos fixos:
a = 3/4
b = 1

# Extremos variáveis da integral da esquerda:
c = lambda x : 0
d = lambda x : (1-x*x)**0.5

n = [6, 8, 10]
for i in n:
  V = 2*np.pi*integral_dupla(f, a, b, c, d, i) # Calcula a integral e multiplica por 2pi, como apresentado na fórmula acima.
  print("V =", V, '\nErro:', abs(V-(11*np.pi/192)), '\n')
  
# (b)  

# A função a ser integrada é a mesma para as duas ordens:
f = lambda x, y : y

# Definimos os extremos fixos:
a = -1
b = 1

# Extremos variáveis da integral da esquerda:
c = lambda x : 0
d = lambda x : np.exp(-x*x)

n = [6, 8, 10]
for i in n:
  V = integral_dupla(f, a, b, c, d, i)*2*np.pi # Calcula a integral e multiplica por 2pi, como apresentado na fórmula acima.
  print("V =", V, '\nErro:', abs(V-3.7582496342318346306), '\n')