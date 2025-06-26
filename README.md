# Urban Disaster Monitor: Um Sistema de Detecção e Classificação de Indivíduos em Cenários de Desastre Urbano Baseado em Visão Computacional

## 📌 Introdução

Eventos de **desastre urbano**, como colapsos estruturais, inundações e deslizamentos de terra, impõem desafios significativos às operações de resposta e resgate. A capacidade de **identificar e categorizar rapidamente** indivíduos como civis ou socorristas em tempo real é crucial para a otimização da alocação de recursos e a minimização de fatalidades. Imagens coletadas por veículos aéreos não tripulados (VANTs), sistemas de vigilância e dispositivos móveis representam uma fonte de dados valiosa para esta finalidade.

Este projeto propõe o desenvolvimento de um **sistema inteligente de visão computacional** para a **detecção e classificação discriminativa de indivíduos (civis e socorristas)** em ambientes impactados por desastres urbanos. Utilizando a arquitetura de **detecção de objetos YOLOv8 (You Only Look Once, versão 8)**, o objetivo é construir uma ferramenta robusta e eficiente capaz de fornecer suporte crítico a autoridades e equipes de emergência durante a fase de resposta a desastres.

Desenvolvido no **SMLab (Smart Metropolis Lab)** do Instituto Metrópole Digital (IMD/UFRN), este trabalho é parte integrante do **Projeto SPICI (Segurança Pública Integrada em Cidades Inteligentes)**. O SPICI visa criar uma plataforma inteligente para coleta, processamento e análise de imagens e informações críticas em tempo real, contribuindo para a gestão de crises e desastres por meio de tecnologias avançadas como visão computacional e inteligência artificial. O `Urban Disaster Monitor` alinha-se diretamente com os objetivos do SPICI ao fornecer uma ferramenta especializada para a identificação de pessoas em cenários de desastre, agregando valor à capacidade de resposta e tomada de decisão da plataforma.

-----

## ⚙️ Metodologia e Materiais

### Aquisição e Anotação de Dados

Para o treinamento e validação do modelo, estamos compilando um **dataset abrangente** composto por diversas imagens que representam cenários típicos de desastres urbanos (e.g., edifícios colapsados, áreas inundadas). As fontes incluem **repositórios públicos de imagens de desastres** e a **geração de imagens sintéticas** para complementar a variabilidade do conjunto de dados. Essa abordagem híbrida visa mitigar a escassez de dados reais de alta qualidade em ambientes de desastre.

A **anotação dos objetos de interesse** está sendo realizada na plataforma **Roboflow**, onde as classes primárias de interesse são definidas como:

  - `People` (Civis): Indivíduos não identificados como membros de equipes de resgate.
  - `Rescuer` (Socorristas): Indivíduos equipados ou identificáveis como parte de equipes de emergência (e.g., bombeiros, paramédicos), frequentemente distinguíveis por uniformes, equipamentos de proteção individual (EPI) ou posturas operacionais.

### Pré-processamento de Dados

As etapas de pré-processamento são cruciais para otimizar o desempenho do modelo e garantir sua generalização:

  - **Redimensionamento**: Todas as imagens são escaladas para dimensões compatíveis com as entradas da arquitetura YOLOv8, balanceando a fidelidade da informação visual com a eficiência computacional.
  - **Aumento de Dados (Data Augmentation)**: Técnicas de aumento de dados (e.g., rotação, translação, espelhamento, ajuste de brilho e contraste, *blurring* e ruído) são aplicadas para aumentar a robustez do modelo, mitigar o risco de *overfitting* e compensar potenciais desequilíbrios entre as classes. A diversificação do dataset através do *data augmentation* é fundamental para simular a variabilidade das condições de imagem em cenários reais de desastre.
  - **Particionamento**: O dataset é dividido em conjuntos de treino, validação e teste, tipicamente nas proporções 70%, 20% e 10%, respectivamente. Essa divisão estratificada assegura uma avaliação imparcial da capacidade de generalização do modelo para dados não vistos.

### Arquitetura do Modelo e Ferramentas

  - **Modelo de Detecção**: **YOLOv8 (You Only Look Once, versão 8)**. Essa arquitetura foi selecionada por sua comprovada eficiência na detecção de objetos em tempo real, combinando alta acurácia com velocidade de inferência, características essenciais para aplicações em cenários emergenciais.
  - **Tarefa**: Detecção de Objetos (Object Detection).
  - **Frameworks e Bibliotecas**: O desenvolvimento e treinamento do modelo são realizados com **Ultralytics YOLOv8**, utilizando **PyTorch** como *backend* de aprendizado profundo, e **OpenCV** para manipulação de imagens e pré-processamento de vídeo.

### Treinamento e Otimização do Modelo

O processo de treinamento envolve as seguintes fases, visando a otimização do desempenho do modelo:

1.  **Conversão de Anotações**: As anotações geradas no formato Roboflow são convertidas para o formato YOLO (`.txt`), que é o padrão de entrada para os modelos da família YOLO.
2.  **Treinamento Iterativo**: O modelo é treinado utilizando um conjunto de parâmetros otimizados, incluindo *batch size*, taxa de aprendizado (*learning rate*) e número de *epochs*. Técnicas de agendamento da taxa de aprendizado (*learning rate schedulers*) e otimizadores (e.g., Adam, SGD) são empregadas para acelerar a convergência e melhorar a performance.
3.  **Validação Contínua**: O desempenho do modelo é monitorado e validado em tempo real durante o treinamento utilizando métricas de desempenho padrão da área em um conjunto de validação separado. Isso permite identificar *overfitting* e ajustar hiperparâmetros de forma dinâmica.
4.  **Testes e Avaliação**: Após o treinamento, o modelo é avaliado extensivamente em um conjunto de teste independente, contendo imagens reais e cenários não vistos, para verificar sua capacidade de generalização e robustez em condições adversas.

-----

## 📊 Análise de Resultados e Métricas

A avaliação da eficácia do modelo será realizada através de um conjunto de métricas quantitativas e qualitativas, fornecendo uma análise abrangente do seu desempenho:

  - **Métricas de Precisão Média (mAP)**:
      - **mAP@0.5**: *Mean Average Precision* calculada com um *IoU (Intersection over Union)* *threshold* de 0.5. Essa métrica é comumente utilizada para avaliação rápida da precisão de detecção.
      - **mAP@0.5:0.95**: *Mean Average Precision* calculada em múltiplos *thresholds* de *IoU*, variando de 0.5 a 0.95 em etapas de 0.05. Essa métrica oferece uma medida mais robusta e abrangente da acurácia de localização e classificação das detecções.
  - **Precisão (Precision) e Recall (Revocação)**: Avaliados por classe para compreender o desempenho individual na identificação de civis e socorristas, indicando a proporção de detecções corretas e a capacidade do modelo de encontrar todas as instâncias relevantes, respectivamente.
  - **Matriz de Confusão (Confusion Matrix)**: Essencial para visualizar e analisar os tipos de erros (falsos positivos, falsos negativos) cometidos pelo modelo, especialmente a confusão entre as classes de interesse, o que é crítico em cenários onde a distinção entre civis e socorristas é vital.
  - **Análise Qualitativa**: Serão apresentadas visualizações das *bounding boxes* e rótulos de classe sobre as imagens de teste. Essa análise visual permite uma avaliação subjetiva da acurácia das detecções e da capacidade do modelo de lidar com variações de escala, oclusão e iluminação.

Os resultados serão detalhadamente apresentados em relatórios técnicos e artigos científicos, incluindo gráficos e tabelas estatísticas, acompanhados de uma análise crítica sobre as limitações do modelo e o potencial de aplicação em contextos reais de desastre.

-----

## 🔍 Discussão e Conclusões (a ser preenchido após a experimentação)

*(Esta seção será dedicada à discussão aprofundada dos resultados obtidos, incluindo a interpretação das métricas de desempenho, a identificação de pontos fortes e fracos do modelo, e a análise de possíveis vieses nos dados. Serão abordadas as limitações inerentes à metodologia, como a sensibilidade a condições de iluminação extremas, oclusão parcial ou total dos indivíduos, e a representatividade do dataset. Além disso, serão propostas futuras linhas de pesquisa para aprimoramento do sistema, como a integração de informações contextuais (e.g., sensores multiespectrais), a otimização para implantação em dispositivos de borda para inferência em tempo real e a exploração de arquiteturas de modelos mais leves. Finalmente, será discutida a integração do `Urban Disaster Monitor` como um módulo crucial dentro da plataforma SPICI, realçando seu impacto potencial na gestão inteligente de desastres e na coordenação de equipes de resgate.)*

-----

## 👥 Equipe do Projeto

O desenvolvimento do `Urban Disaster Monitor` é realizado por alunos da matéria de Visão Computacional, dada pelo professor **Helton Maia** da ECT/UFRN:

| Membro              | Papel               |
| :------------------ | :------------------ |
| **jagaldino** | Pesquisador(a)      |
| **MariaCarolinass** | Pesquisador(a)      |
| **heltonmaia** | Professor Orientador |

-----

## 💻 Instruções para Reprodução Local

Para replicar o ambiente de desenvolvimento e executar o modelo localmente, siga os passos abaixo. Certifique-se de que o sistema atenda aos requisitos de hardware para treinamento de modelos de *deep learning* (e.g., GPU com CUDA).

```bash
# 1. Clonar o repositório do projeto
git clone https://github.com/seu-usuario/urban-disaster-monitor.git
cd urban-disaster-monitor

# 2. Instalar as dependências do projeto (requer Python 3.8+ e pip)
# É altamente recomendado o uso de um ambiente virtual (e.g., conda ou venv)
pip install -r requirements.txt

# 3. Exemplo de comando para treinamento do modelo utilizando a CLI da Ultralytics
# Certifique-se de que o arquivo 'dataset.yaml' (com as configurações do seu dataset)
# e o modelo base 'yolov8n.pt' (ou outra variante YOLOv8) estejam configurados corretamente.
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640
```

-----

## 🧠 Repositório de Dados

O dataset anotado e pré-processado será disponibilizado para a comunidade científica (ou referenciado através de um link permanente) após a conclusão das etapas de anotação e validação, aderindo aos princípios de ciência aberta e reprodutibilidade.

  - **Link Roboflow**: *(a ser adicionado)*
  - **Formato das Anotações**: YOLOv8 (arquivos `.txt` contendo as coordenadas das *bounding boxes* e os IDs das classes, acompanhados das imagens correspondentes, seguindo a convenção de nomenclatura do YOLO).

-----

## 📁 Estrutura do Projeto

```
EM CONSTRUÇÃO
```

-----

## 📚 Referências Bibliográficas

  - Projeto SPICI (Smart Platform for Images and Critical Information): [https://smlab.imd.ufrn.br/projeto-spici/](https://smlab.imd.ufrn.br/projeto-spici/)
  - Ultralytics YOLOv8 Docs: [https://docs.ultralytics.com](https://docs.ultralytics.com)
  - Roboflow: [https://roboflow.com](https://roboflow.com)
  - Redmon, J., Farhadi, A. (2018). **YOLOv3: An Incremental Improvement**. *arXiv preprint arXiv:1804.02767*.
  - Redmon, J., Divvala, S., Girshick, R., Farhadi, A. (2016). **You Only Look Once: Unified, Real-Time Object Detection**. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 779-788.
  - Dataset públicos: [Google Open Images](https://storage.googleapis.com/openimages/web/index.html), [Kaggle Datasets](https://www.kaggle.com/datasets)
