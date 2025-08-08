import math
from scipy.optimize import fsolve
import argparse

# --- Funciones de cálculo termodinámico ---

def antoine_pressure(A, B, C, T):
    """
    Calcula la presión de saturación usando la ecuación de Antoine.
    T debe estar en grados Celsius para esta implementación de Antoine.
    """
    # log10(P0) = A - B / (T + C)
    # P0 = 10^(A - B / (T + C))
    if T + C == 0:
        raise ValueError("La temperatura (T) más la constante C de Antoine no pueden ser cero.")
    return 10**(A - B / (T + C))

def van_laar_activity_coefficients(x_A, A12, A21):
    """
    Calcula los coeficientes de actividad usando el modelo de Van Laar.
    x_A: fracción molar del componente A en la fase líquida.
    A12, A21: parámetros del modelo de Van Laar.
    """
    if not (0 <= x_A <= 1):
        raise ValueError("La fracción molar x_A debe estar entre 0 y 1.")

    x_B = 1 - x_A

    if x_A == 0:
        ln_gamma_A = A12
        ln_gamma_B = 0
    elif x_B == 0:
        ln_gamma_A = 0
        ln_gamma_B = A21
    else:
        denominator = (A12 * x_A + A21 * x_B)
        if denominator == 0:
            raise ValueError("Denominador cero en el cálculo de Van Laar. Revise los parámetros A12, A21 y x_A.")
        term_A = (A12 * A21 * x_B) / denominator**2
        term_B = (A12 * A21 * x_A) / denominator**2

        ln_gamma_A = A12 * term_A
        ln_gamma_B = A21 * term_B

    gamma_A = math.exp(ln_gamma_A)
    gamma_B = math.exp(ln_gamma_B)

    return gamma_A, gamma_B

def isothermal_vle(x_A, T, antoine_A, antoine_B, van_laar_A12, van_laar_A21):
    """
    Calcula el equilibrio líquido-vapor isotérmico (P_T y y_A).
    """
    gamma_A, gamma_B = van_laar_activity_coefficients(x_A, van_laar_A12, van_laar_A21)

    P0_A = antoine_pressure(antoine_A[0], antoine_A[1], antoine_A[2], T)
    P0_B = antoine_pressure(antoine_B[0], antoine_B[1], antoine_B[2], T)

    P_T = x_A * gamma_A * P0_A + (1 - x_A) * gamma_B * P0_B

    y_A = (x_A * gamma_A * P0_A) / P_T

    return P_T, y_A, gamma_A, gamma_B

def isobaric_vle(x_A, P_T, antoine_A, antoine_B, van_laar_A12, van_laar_A21):
    """
    Calcula el equilibrio líquido-vapor isobárico (T y y_A).
    """
    def objective_function(T_guess):
        try:
            gamma_A, gamma_B = van_laar_activity_coefficients(x_A, van_laar_A12, van_laar_A21)
            P0_A = antoine_pressure(antoine_A[0], antoine_A[1], antoine_A[2], T_guess)
            P0_B = antoine_pressure(antoine_B[0], antoine_B[1], antoine_B[2], T_guess)
            return (x_A * gamma_A * P0_A / P_T) + ((1 - x_A) * gamma_B * P0_B / P_T) - 1
        except ValueError:
            return 1e10 # Retorna un valor grande para evitar soluciones inválidas

    # Intentar con una suposición inicial de temperatura, por ejemplo, 50 grados Celsius
    # Se puede mejorar la suposición inicial si se conocen los rangos de temperatura
    T_solution = fsolve(objective_function, 50)[0]

    gamma_A, gamma_B = van_laar_activity_coefficients(x_A, van_laar_A12, van_laar_A21)
    P0_A = antoine_pressure(antoine_A[0], antoine_A[1], antoine_A[2], T_solution)
    P0_B = antoine_pressure(antoine_B[0], antoine_B[1], antoine_B[2], T_solution)

    y_A = (x_A * gamma_A * P0_A) / P_T

    return T_solution, y_A, gamma_A, gamma_B

# --- Interfaz de usuario y lógica principal ---

def get_user_input_interactive():
    print("\n--- Calculadora de Equilibrio Líquido-Vapor ---")

    while True:
        try:
            x_A = float(input("Ingrese la fracción molar líquida del componente A (x_A): "))
            if not (0 <= x_A <= 1):
                raise ValueError("x_A debe estar entre 0 y 1.")
            break
        except ValueError as e:
            print(f"Error: {e}. Intente de nuevo.")

    # Constantes de Antoine para el componente A (ej: [A, B, C])
    print("\nIngrese las constantes de Antoine para el componente A (A, B, C):")
    while True:
        try:
            antoine_A = [float(input("A_A: ")), float(input("B_A: ")), float(input("C_A: "))]
            break
        except ValueError:
            print("Entrada inválida. Por favor, ingrese números para las constantes de Antoine.")

    # Constantes de Antoine para el componente B (ej: [A, B, C])
    print("\nIngrese las constantes de Antoine para el componente B (A, B, C):")
    while True:
        try:
            antoine_B = [float(input("A_B: ")), float(input("B_B: ")), float(input("C_B: "))]
            break
        except ValueError:
            print("Entrada inválida. Por favor, ingrese números para las constantes de Antoine.")

    # Parámetros del modelo de Van Laar
    print("\nIngrese los parámetros del modelo de Van Laar (A12, A21):")
    while True:
        try:
            van_laar_A12 = float(input("A12: "))
            van_laar_A21 = float(input("A21: "))
            break
        except ValueError:
            print("Entrada inválida. Por favor, ingrese números para los parámetros de Van Laar.")

    while True:
        eq_type = input("\nTipo de equilibrio a calcular (isotermico/isobarico): ").lower()
        if eq_type in ["isotermico", "isobarico"]:
            break
        else:
            print("Tipo de equilibrio no válido. Por favor, elija 'isotermico' o 'isobarico'.")

    P_T = None
    T = None

    if eq_type == "isotermico":
        while True:
            try:
                T = float(input("Ingrese la temperatura del sistema en grados Celsius (T): "))
                break
            except ValueError:
                print("Entrada inválida. Por favor, ingrese un número para la temperatura.")
    elif eq_type == "isobarico":
        while True:
            try:
                P_T = float(input("Ingrese la presión total del sistema (P_T): "))
                break
            except ValueError:
                print("Entrada inválida. Por favor, ingrese un número para la presión.")

    while True:
        partially_miscible_input = input("\n¿La mezcla es parcialmente miscible? (s/n): ").lower()
        if partially_miscible_input in ['s', 'n']:
            partially_miscible = partially_miscible_input == 's'
            break
        else:
            print("Entrada inválida. Por favor, ingrese 's' o 'n'.")

    return {
        "x_A": x_A,
        "P_T": P_T,
        "T": T,
        "antoine_A": antoine_A,
        "antoine_B": antoine_B,
        "van_laar_A12": van_laar_A12,
        "van_laar_A21": van_laar_A21,
        "eq_type": eq_type,
        "partially_miscible": partially_miscible
    }

def save_results(filename, results):
    with open(filename, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(f"Resultados guardados en {filename}")

def main():
    parser = argparse.ArgumentParser(description="Calculadora de Equilibrio Líquido-Vapor para mezclas binarias no ideales.")
    parser.add_argument('--interactive', action='store_true', help='Ejecutar en modo interactivo.')
    parser.add_argument('--xA', type=float, help='Fracción molar líquida del componente A.')
    parser.add_argument('--PT', type=float, help='Presión total del sistema (para equilibrio isobárico).')
    parser.add_argument('--T', type=float, help='Temperatura del sistema en grados Celsius (para equilibrio isotérmico).')
    parser.add_argument('--antoineA', nargs=3, type=float, help='Constantes de Antoine para el componente A (A, B, C).')
    parser.add_argument('--antoineB', nargs=3, type=float, help='Constantes de Antoine para el componente B (A, B, C).')
    parser.add_argument('--vanLaarA12', type=float, help='Parámetro A12 del modelo de Van Laar.')
    parser.add_argument('--vanLaarA21', type=float, help='Parámetro A21 del modelo de Van Laar.')
    parser.add_argument('--eqType', type=str, choices=['isotermico', 'isobarico'], help='Tipo de equilibrio a calcular (isotermico/isobarico).')
    parser.add_argument('--partiallyMiscible', type=str, choices=['s', 'n'], help='¿La mezcla es parcialmente miscible? (s/n).')
    parser.add_argument('--outputFile', type=str, help='Nombre del archivo para guardar los resultados.')

    args = parser.parse_args()

    if args.interactive:
        user_data = get_user_input_interactive()
        if not user_data:
            return
    else:
        # Validar que todos los argumentos necesarios estén presentes para el modo no interactivo
        required_args = ['xA', 'antoineA', 'antoineB', 'vanLaarA12', 'vanLaarA21', 'eqType']
        if args.eqType == 'isotermico':
            required_args.append('T')
        elif args.eqType == 'isobarico':
            required_args.append('PT')

        for arg_name in required_args:
            if getattr(args, arg_name) is None:
                parser.error(f"El argumento --{arg_name} es requerido en modo no interactivo para el tipo de equilibrio {args.eqType}.")

        user_data = {
            "x_A": args.xA,
            "P_T": args.PT,
            "T": args.T,
            "antoine_A": args.antoineA,
            "antoine_B": args.antoineB,
            "van_laar_A12": args.vanLaarA12,
            "van_laar_A21": args.vanLaarA21,
            "eq_type": args.eqType,
            "partially_miscible": args.partiallyMiscible == 's' if args.partiallyMiscible else False
        }

    x_A = user_data["x_A"]
    P_T = user_data["P_T"]
    T = user_data["T"]
    antoine_A = user_data["antoine_A"]
    antoine_B = user_data["antoine_B"]
    van_laar_A12 = user_data["van_laar_A12"]
    van_laar_A21 = user_data["van_laar_A21"]
    eq_type = user_data["eq_type"]
    partially_miscible = user_data["partially_miscible"]

    results = {}
    print("\n--- Resultados ---")

    try:
        if eq_type == "isotermico":
            P_T_calc, y_A, gamma_A, gamma_B = isothermal_vle(x_A, T, antoine_A, antoine_B, van_laar_A12, van_laar_A21)
            results["Temperatura (T)"] = f"{T:.2f} °C"
            results["Presión total (P_T)"] = f"{P_T_calc:.4f}"
        elif eq_type == "isobarico":
            T_calc, y_A, gamma_A, gamma_B = isobaric_vle(x_A, P_T, antoine_A, antoine_B, van_laar_A12, van_laar_A21)
            results["Presión total (P_T)"] = f"{P_T:.2f}"
            results["Temperatura (T)"] = f"{T_calc:.2f} °C"

        results["Fracción molar líquida de A (x_A)"] = f"{x_A:.4f}"
        results["Fracción molar vapor de A (y_A)"] = f"{y_A:.4f}"
        results["Coeficiente de actividad gamma_A"] = f"{gamma_A:.4f}"
        results["Coeficiente de actividad gamma_B"] = f"{gamma_B:.4f}"

        for key, value in results.items():
            print(f"{key}: {value}")

        # Validación para mezclas parcialmente miscibles
        if partially_miscible:
            # Un criterio más robusto para miscibilidad parcial podría implicar el cálculo de los puntos azeotrópicos
            # o el análisis de la estabilidad termodinámica (e.g., dG_mixing < 0).
            # Para una simplificación, se puede usar un umbral de gamma alto como indicador.
            if gamma_A > 5 or gamma_B > 5: # Umbral ajustado, puede ser configurable
                print("\n¡Advertencia! Coeficientes de actividad muy altos. Esto podría indicar separación de fases líquidas o inestabilidad del sistema.")
                print("Considere la posibilidad de formación de dos fases líquidas bajo estas condiciones.")

        if args.outputFile:
            save_results(args.outputFile, results)

    except ValueError as e:
        print(f"Error en el cálculo: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()