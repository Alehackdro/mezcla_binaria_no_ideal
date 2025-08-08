import math
from scipy.optimize import fsolve


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

# --- Funciones de conversión de unidades ---

def convert_temperature(value, unit_from, unit_to):
    if unit_from.lower() == unit_to.lower():
        return value

    # Convertir a Celsius primero
    if unit_from.lower() == 'k':
        celsius = value - 273.15
    elif unit_from.lower() == 'f':
        celsius = (value - 32) * 5/9
    elif unit_from.lower() == 'c':
        celsius = value
    else:
        raise ValueError(f"Unidad de temperatura desconocida: {unit_from}. Unidades soportadas: C, K, F")

    # Convertir de Celsius a la unidad deseada
    if unit_to.lower() == 'k':
        return celsius + 273.15
    elif unit_to.lower() == 'f':
        return (celsius * 9/5) + 32
    elif unit_to.lower() == 'c':
        return celsius
    else:
        raise ValueError(f"Unidad de temperatura desconocida: {unit_to}. Unidades soportadas: C, K, F")

def convert_pressure(value, unit_from, unit_to):
    if unit_from.lower() == unit_to.lower():
        return value

    # Convertir a kPa primero (asumiendo kPa como base para Antoine si no se especifica)
    if unit_from.lower() == 'atm':
        kpa = value * 101.325
    elif unit_from.lower() == 'mmhg':
        kpa = value * 0.133322
    elif unit_from.lower() == 'psi':
        kpa = value * 6.89476
    elif unit_from.lower() == 'kpa':
        kpa = value
    else:
        raise ValueError(f"Unidad de presión desconocida: {unit_from}. Unidades soportadas: kPa, atm, mmHg, psi")

    # Convertir de kPa a la unidad deseada
    if unit_to.lower() == 'atm':
        return kpa / 101.325
    elif unit_to.lower() == 'mmhg':
        return kpa / 0.133322
    elif unit_to.lower() == 'psi':
        return kpa / 6.89476
    elif unit_to.lower() == 'kpa':
        return kpa
    else:
        raise ValueError(f"Unidad de presión desconocida: {unit_to}. Unidades soportadas: kPa, atm, mmHg, psi")

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
                temp_input = input("Ingrese la temperatura del sistema (ej. 25 C, 298.15 K, 77 F): ").split()
                value = float(temp_input[0])
                unit = temp_input[1].upper() if len(temp_input) > 1 else 'C' # Default a Celsius
                T = convert_temperature(value, unit, 'C') # Convertir a Celsius para Antoine
                break
            except (ValueError, IndexError) as e:
                print(f"Entrada inválida. Por favor, ingrese un número y una unidad válida (C, K, F). Error: {e}")
    elif eq_type == "isobarico":
        while True:
            try:
                pressure_input = input("Ingrese la presión total del sistema (ej. 101.325 kPa, 1 atm, 760 mmHg, 14.7 psi): ").split()
                value = float(pressure_input[0])
                unit = pressure_input[1].lower() if len(pressure_input) > 1 else 'kpa' # Default a kPa
                P_T = convert_pressure(value, unit, 'kpa') # Convertir a kPa (o la unidad base de tus constantes Antoine)
                break
            except (ValueError, IndexError) as e:
                print(f"Entrada inválida. Por favor, ingrese un número y una unidad válida (kPa, atm, mmHg, psi). Error: {e}")

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

def main():
    user_data = get_user_input_interactive()
    if not user_data:
        return

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

    except ValueError as e:
        print(f"Error en el cálculo: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()