"""
Tarefa 3 - Modelagem de um Sistema de Resfriamento de Chips - MAP3121
Data de entrega: 10/07/2022

    Rodrigo Gebara Reis - NUSP: 11819880
    Victor Rocha da Silva - NUSP: 11223782

"""

# Import das funções do EP1 e EP2:

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

def integral_a_to_b(f, a, b, n):
# Entradas: função f(x), extremos a, b do intervalo de integração, número de nós n
# Saída: resultado da integral de a até b de f(x)
# Caso mais geral da função integral(f, n), aplica a fórmula de mudança de variável deduzida acima

  l = (b-a)/2 # Define-se dois parâmetros para a transformação linear: metade do tamanho do intervalo
  m = (a+b)/2 # e o ponto médio, segundo as fórmulas deduzidas

  y, w = nos_pesos(n) # Recupera os valores fornecidos no arquivo .txt

  x = l*y + m # Aplica a transformação linear do intervalo [-1,1] para [a,b]
  sum = 0
   
  for i in range(n): # Realiza a soma descrita no algoritmo deduzido (soma de w_j*f(x_j))
    sum += w[i]*f(x[i])

  sum = sum*l # Finaliza a aplicação da fórmula geral para [a,b] multiplicando por (b-a)/2

  return sum # Retorna o valor final da soma, aproximadamente a integral definida


def decompLU(a, b, c):
  n = a.shape[0]
  l, u = np.zeros(n), np.zeros(n) # Inicia os vetores l e u

  l[0] = 0 # Assim como a[0] = 0, definimos l[0] = 0
  u[0] = b[0]
  
  # Segue as formulações definidas no texto acima
  for i in range(1, n):
    l[i] = a[i]/u[i-1]
    u[i] = b[i] - l[i]*c[i-1]
    
  return(l, u)

def resolveSistemaTridiagonal(a, b, c, d):
  n = a.shape[0]
  # Decompõe a matriz tridiagonal em uma triangular inferior, caracterizada pelo vetor l,
  # e uma triangular superior, caracterizada pelos vetores u e c
  l, u = decompLU(a, b, c)

  #Ly = d
  y = np.zeros(n)

  y[0] = d[0]
  for i in range(1, n):
    y[i] = d[i] - l[i]*y[i-1]

  #Ux = y
  x = np.zeros(n)

  x[n-1] = y[n-1]/u[n-1]
  for i in range(n-2, -1, -1):
    x[i] = (y[i]-c[i]*x[i+1])/u[i]

  return x

import numpy as np

#-----------------------------------------------------------------------------#

# Rotinas do EP3:
    
    
# Primeiramente, cria-se uma rotina para determinar todos os pontos igualmente 
# espaçados dentro do intervalo [0,1], de acordo com o apresentado anteriormente: x_i = ih, h = 1/(n+1):
def pontos(n): # Recebe apenas o número n de pontos desejados, sem contar os extremos
  return [i/(n+1) for i in range(0, n+2)] # Retorna o conjunto de pontos x_i no intervalo [0,1]

# Função para calcular o vetor de produtos internos <f, phi> no intervalo [0,1]
def fxphi(f, n): # Recebe a função f e o número de pontos n
  x = pontos(n) # x_i dentro de [0,1] para as funções chapéu e integrais
  h = 1/(n+1)

  prod = []

  for i in range(1,len(x)-1): # Calcula-se cada linha (<f, phi_i>) do vetor de produtos internos
    g1 = lambda y : (y-x[i-1])/h*f(y)
    g2 = lambda y : (x[i+1]-y)/h*f(y)

    first = integral_a_to_b(g1, x[i-1], x[i], 10) # Calcula a primeira integral, referente ao intervalo [x[i-1],x[i]] 
    second = integral_a_to_b(g2, x[i], x[i+1], 10) # Calcula a segunda integral, referente ao intervalo [x[i],x[i+1]]

    prod.append(first + second)
    
  return np.array(prod)

def phi_i(point, x, i): # Definição das funções phi_i, de acordo com a definição apresentada na seção 3.2
# Recebe o ponto 'point', o conjunto de pontos x, e o índice para definição do intervalo [x[i-1], x[i+1]]
  result = 0
  if (point >= x[i-1] and point < x[i]):
    result = (point-x[i-1])/(x[i]-x[i-1])

  if (point >= x[i] and point <= x[i+1]):
    result = (x[i+1]-point)/(x[i+1]-x[i])

  return result # Retorna o valor de phi_i para um ponto 'point'

def vector_a(n): # Define o vetor da diagonal abaixo da principal, completo com -1/h, recebe a ordem n da matriz tridiagonal
  a = [0]
  h = 1/(n+1)
  for i in range(n-1):
    a.append(-1/h)

  return np.array(a)

def vector_b(n): # Define o vetor da diagonal principal, completo com 2/h, recebe a ordem n da matriz tridiagonal
  b = []
  h = 1/(n+1)
  for i in range(n):
    b.append(2/h)

  return np.array(b)

def vector_c(n): # Define o vetor da diagonal acima da principal, completo com -1/h, recebe a ordem n da matriz tridiagonal
  c = []
  h = 1/(n+1)
  for i in range(n-1):
    c.append(-1/h)
  c.append(0)

  return np.array(c)

def u(f, point, n): # Define a função u_barra para um ponto arbitrário 'point'
# Recebe a função f, o ponto 'point' e o número de pontos n
  x = pontos(n) # Define os n nós para os splines
  sol = np.array(resolveSistemaTridiagonal(vector_a(n), vector_b(n), vector_c(n), fxphi(f, n))) # Resolve o sistema tridiagonal deduzido

  phi = []
  for i in range(1,len(x)-1):
    phi.append(phi_i(point,x,i)) # Armazena os valores de cada phi_i em um vetor

  phi = np.array(phi)

  return sol@phi # Retorna o valor de u_barra em um ponto arbitrário 'point'

def pontos_L(L, n): # Recebe comprimento L e número de pontos n. Define os nós no intervalo [0,L]
  return L*np.array(pontos(n)) # Retorna os nós do intervalo [0,1] multiplicados por L

def fxphi_L(f, L, n): # Recebe a função f, o tamanho do intervalo L, e o número de pontos n
  x = pontos_L(L, n) # x_i dentro de [0,L] para as funções chapéu e integrais
  h = L/(n+1) # Define L no caso mais geral

  prod = []

  for i in range(1,len(x)-1): # Calcula-se cada linha (<f, phi_i>) do vetor de produtos internos
    g1 = lambda y : (y-x[i-1])/h*f(y)
    g2 = lambda y : (x[i+1]-y)/h*f(y)

    first = integral_a_to_b(g1, x[i-1], x[i], 10) # Calcula a primeira integral, referente ao intervalo [x[i-1],x[i]] 
    second = integral_a_to_b(g2, x[i], x[i+1], 10) # Calcula a segunda integral, referente ao intervalo [x[i],x[i+1]]

    prod.append(first + second)
    
  return np.array(prod) # Devolve os valores numéricos referentes aos produtos internos

def phixphi_geral(L, k, q, i, j, n, x): # Recebe o comprimento L, as funções k, q, conjunto de nós x, valores de i e j, número de nós n
  h = L/(n+1) # Define h, distância entre dois nós
  result = 0

  if i == j: # Aplica a integral geral para k != 0 e q != 0, i = j
    function1 = lambda t : k(t)+q(t)*(t-x[i-1])**2
    function2 = lambda t : k(t)+q(t)*(t-x[i+1])**2
    result = (1/h**2)*(integral_a_to_b(function1, x[i-1], x[i], 10)+integral_a_to_b(function2, x[i], x[i+1], 10))

  if j - i == 1: # Aplica a integral geral para k != 0 e q != 0, j-i=1
    function = lambda t : k(t)+q(t)*(t-x[i])*(t-x[i+1])
    result = (-1/h**2)*integral_a_to_b(function, x[i], x[i+1], 10)

  return result # Devolve o produto interno entre phi_i e phi_j

def u_L(f, k, q, L, point, n): # Define a função u_barra para um ponto arbitrário 'point' em um intervalo [0,L]
# Recebe as funções f, k, q, o comprimento L, o ponto 'point' e o número de pontos n
  x = pontos_L(L, n) # Define os nós no intervalo [0,L]
  # Define os vetores da matriz tridiagonal para encontrar alpha
  a = [0]
  b = []
  c = []
  d = fxphi_L(f, L, n) 

  i = 0
  for j in range(1, len(x)-1):
    dot_prod = phixphi_geral(L, k, q, i+1, j, n, x) # Preenche a diagonal principal com os produtos internos <phi_i, phi_i>
    b.append(dot_prod)

    if i >= 1:
      dot_prod = phixphi_geral(L, k, q, i, j, n, x) # Preenche as diagonais secundárias com os produtos internos <phi_i, phi_j>
      # As duas são iguais, fora o termo zero necessário para a matriz tridiagonal
      a.append(dot_prod)
      c.append(dot_prod)
    i += 1 

  a = np.array(a)
  b = np.array(b)
  c.append(0)
  c = np.array(c)

  sol = np.array(resolveSistemaTridiagonal(a, b, c, d)) # Resolve o sistema tridiagonal, define alpha

  phi = []
  for i in range(1,len(x)-1):
    phi.append(phi_i(point,x,i)) # Armazena os valores de cada phi_i em um vetor

  phi = np.array(phi)

  return sol@phi # Retorna o valor de u_barra em um ponto arbitrário 'point'

def u_full(f, k, k_prime, q, a, b, L, point, n): # Define a solução mais geral possível: condições não-homogêneas e intervalo [0,L]
# Recebe funções f, k, k_prime, q, condições de borda a, b, comprimento L, ponto 'point', número de nós n
# Aplica as deduções apresentadas
  f_mod = lambda t : f(t) + (b-a)*k_prime(t) - q(t)*(a+(b-a)*t)

  v = lambda x : u_L(f_mod, k, q, L, x, n)
  u_tot = lambda x : v(x) + a + (b-a)*x/L

  return u_tot(point)

if __name__ == "__main__":
    print("Escolha alguma das funções e execute com base no arquivo LEIAME.txt!")