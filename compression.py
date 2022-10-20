import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np
import math
from PIL import Image

# Parâmetros
direcoes = [1, -1] # Direções possíveis para a realização do espelhamento
angs = [0, 90, 180, 270] # Angulos possíveis para a realização da rotação


path_orig = 'Original_Images/' # Caminho em que a imagem original está
path_dest = 'Compressed_Images/' # Caminho em que a imagem descomprimida será salva

# Retorna a imagem em escala de cinza
def retorna_greyscale(img):
    return np.mean(img[:,:,:2], 2)

# Tranformações lineares
def reduzir(img, fator):
    resultado = np.zeros((img.shape[0] // fator, img.shape[1] // fator)) # Cria a matriz vazio do tamanho necessário
    for i in range(resultado.shape[0]):
        for j in range(resultado.shape[1]):
            resultado[i,j] = np.mean(img[i*fator:(i+1)*fator,j*fator:(j+1)*fator]) # Faz a redução da imagem
    return resultado # Retorna a imagem reduzida pelo fator

# Rotaciona a imagem em função do angulo passado como argumento
def rotacionar(img, ang):
    return ndimage.rotate(img, ang, reshape=False)

# Espelha as imagens em relação a dimensão passada como argumento
def espelhar(img, direcao):
    return img[::direcao,:]

# Realiza todas as operações geométricas lineares
def aplica_transf(img, direcao, ang, contraste=1.0, brilho=0.0):
    return contraste * rotacionar(espelhar(img, direcao), ang) + brilho

# Encontra contraste e brilho
def contraste_e_brilho(D, S):
    # Ajusta o contraste e o brilho
    A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
    b = np.reshape(D, (D.size,))
    x, _, _, _ = np.linalg.lstsq(A, b)
    return x[1], x[0]

# Gera os blocos de transformação
def gera_blocos_transformacao(img, tam_orig, tam_dest, step):
    fator = tam_orig // tam_dest # Calcula o fator de escala
    blocos_transformados = []  # Inicializa matriz dos blocos
    for k in range((img.shape[0] - tam_orig) // step + 1):
        for l in range((img.shape[1] - tam_orig) // step + 1):
            # Extrai o bloco de origem e faz a redução para o shape do bloco de destino
            S = reduzir(img[k*step:k*step+tam_orig,l*step:l*step+tam_orig], fator)
            # Gera todos os possíveis blocos de transformação
            for direcao in direcoes: # Para cada direção
                for ang in angs: # Para cada angulo
                    # Gera as transformações
                    blocos_transformados.append((k, l, direcao, ang, aplica_transf(S, direcao, ang)))
    return blocos_transformados

def comprimir(img, tam_orig, tam_dest, step):
    transformacoes = [] # Inicializa matriz que armazenará as transformações
    blocos_transformados = gera_blocos_transformacao(img, tam_orig, tam_dest, step) # Gera todas as transformações
    max_i = img.shape[0] // tam_dest # Encontra o tamanho necessário em x
    max_j = img.shape[1] // tam_dest # Encontra o tamanho necessário em y
    for i in range(max_i):
        transformacoes.append([])
        for j in range(max_j):
            print("{}/{} ; {}/{}".format(i, max_i, j, max_j)) # Printa o progresso da compressão
            transformacoes[i].append(None)
            min_d = float('inf') # Configura o mínimo como infinito
            # Extrai o bloco de transformaçao
            D = img[i*tam_dest:(i+1)*tam_dest,j*tam_dest:(j+1)*tam_dest]
            # Pega todas as transformações e escolhe a melhor
            for k, l, direcao, ang, S in blocos_transformados:
                contraste, brilho = contraste_e_brilho(D, S) # Encontra o constraste e o brilho
                S = contraste*S + brilho # Aplica o constraste e o brilho
                d = np.sum(np.square(D - S)) # Calcula o erro quadrático médio
                if d < min_d: # Verifica se o EQM é menor que o mínimo
                    min_d = d # Atualiza o mínimo
                    transformacoes[i][j] = (k, l, direcao, ang, contraste, brilho) # Adiciona a transformação na matriz
    return transformacoes # Retorna a matriz de transformações

def descomprimir(transformacoes, tam_orig, tam_dest, step, nb_iter=8):
    fator = tam_orig // tam_dest # Encontra o fator de redução
    altura = len(transformacoes) * tam_dest # Encontra qual deve ser a altura da imagem
    largura = len(transformacoes) * tam_dest # Encontra qual deve ser a largura da imagem

    iteracoes = [np.random.randint(0, 256, (altura, largura))] # Gera a matriz com elementos aleatórios
    # iteracoes = [np.ones((altura, largura))*255] # Gera a matriz com um
    # iteracoes = [np.zeros((altura, largura))] # Gera a matriz com zeros
    # iteracoes = [np.ones((altura, largura))*128] # Gera a matriz com todos os bits com cinza intermediário

    img_atual = np.zeros((altura, largura)) 
    for i_iter in range(nb_iter): # Para cada iteração
        print(i_iter)
        for i in range(len(transformacoes)):
            for j in range(len(transformacoes[i])):
                k, l, flip, ang, contraste, brilho = transformacoes[i][j] # identifica os parametros das transformações
                S = reduzir(iteracoes[-1][k*step:k*step+tam_orig,l*step:l*step+tam_orig], fator) # Reduz o bloco
                D = aplica_transf(S, flip, ang, contraste, brilho) # Aplica as transformações
                img_atual[i*tam_dest:(i+1)*tam_dest,j*tam_dest:(j+1)*tam_dest] = D
        iteracoes.append(img_atual) # Armazena o resultado
        img_atual = np.zeros((altura, largura)) # Reseta a matriz
    Image.fromarray(iteracoes[nb_iter-1]).save(path_dest+name_image)
    return iteracoes  # Retorna as imagens descomprimidas para cada uma das iterações

def plot_iteracoes(iteracoes,imagem):
    # Configura o plot
    plt.figure('Iterações') # Nova figura
    n_linhas = math.ceil(np.sqrt(len(iteracoes))) # Encontra o número necessário de linhas para os plots
    n_colunas = n_linhas # Encontra o número necessário de colunas para os plots
    for i, img in enumerate(iteracoes): # Plota todas as interações
        plt.subplot(n_linhas, n_colunas, i+1) # Define o lugar em que será plotado a imagem
        plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='none') # Plota a imagem 
        plt.title(str(i) + ' (' + '{0:.2f}'.format(np.sqrt(np.mean(np.square(imagem - img)))) + ')')
        # Exibe o número da iteração e a raiz quadrado do erro quadrático médio
    plt.tight_layout() # Ajusta o plot para não ficar bagunçado
    plt.figure('Imagem com o maxímo de iterações definidas') # Nova figura
    plt.imshow(iteracoes[len(iteracoes)-1], cmap='gray', vmin=0, vmax=255, interpolation='none') # Plota a imagem 

tam_orig = 8 # Tamanho dos blocos de origem
tam_dest = 4 # Tamanho dos blocos de destino
step = 8 
n_iteracoes = 8 # número de iterações

# Imagens
# name_image = 'monkey.gif'
name_image = 'flower.gif'
# name_image = 'lena.gif'
              
if __name__ == '__main__':
    img = mpimg.imread(path_orig + name_image) # Faz a leitura da imagem que será comprimida
    img = retorna_greyscale(img) # Retorna a imagem em greyscale
    img = reduzir(img, 4) # Reduz a imagem
    plt.figure('Imagem original') # Nova figura
    plt.imshow(img, cmap='gray', interpolation='none') # Plota a imagem original
    transformacoes = comprimir(img, tam_orig, tam_dest, step) # Comprime a imagem
    iteracoes = descomprimir(transformacoes, tam_orig, tam_dest, step, n_iteracoes) # Descomprime a imagem
    plot_iteracoes(iteracoes,img) # Plota as imagens em cada uma das iterações
    plt.show() # Mostras as imagens