# CT-Near

## Nova Near Real Time

Essa aplicação implementa os processos de extração, filtragem e interpolações
dos dados observados de estações meteorologicas.

## Funcionamento

Na sua execução, inicialmente a aplicação faz um query ao BQ.
Na sequencia aplica uma serie de filtros, para validação dos dados, checando
continueidade temporal, monotonia da serie, aceleração da serie, e tambem os 
validadando espacialmente, comparando contra vizinhos proximos, levando em 
consideração não distancia, mas tambem diferenças em altitudes.

## Operação

### Instalação

para instalar esta aplicação é necessario construir um ambiente virtual python3.8
ou mais recente.

```sh
python3.8 -m venv venv --prompt=ct-near
source venv/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

### Execução

```sh
source venv/bin/activate
python3 main.py --help
```

Ha uma peculiaridade na execução das variaveis de vento. Devido a forma como são
feitos os processos de filtragem e interpolação, as variaveis resultantes, tanto
no bq quanto no netCDF4 são diferentes entre si. O bq recebe velocidade e direção
vento (stations_verified.wind_speed, stations_verified.wind_dir) e o netCDF4 gera
dois arquivos, um para componente U e outro para componente V (10m_u_component_of_wind, 
10m_v_component_of_wind). Devido a essa peculiaridade a execução para a variavel
vento usa uma variavel de controle que viola o padrão estabelicido nas demais
variaveis, utilizando a variavel *10m_wind*, sem especificar componentes ou tipo.
Esta variavel deve ser utilizada tambem no path do arquivo de saida, assim o codigo
entende qual variavel deve substituir, para produzir os resultados corretos.
