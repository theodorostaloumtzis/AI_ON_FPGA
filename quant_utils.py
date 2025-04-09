import re

def extract_quant_config(model, verbose=True):
    quant_config = {}

    for layer in model.layers:
        layer_name = layer.name
        quant_config[layer_name] = {}

        # Handle kernel quantizer
        kq = getattr(layer, 'kernel_quantizer', None)
        if kq:
            if isinstance(kq, str):
                match = re.search(r'quantized_bits\((\d+),\s*(\d+)', kq)
                if match:
                    bits, integer = int(match.group(1)), int(match.group(2))
                    quant_config[layer_name]['weights'] = {
                        'bitwidth': bits,
                        'integer_bits': integer,
                        'signed': True
                    }
            elif hasattr(kq, 'bits') and hasattr(kq, 'integer'):
                quant_config[layer_name]['weights'] = {
                    'bitwidth': kq.bits,
                    'integer_bits': kq.integer,
                    'signed': getattr(kq, 'signed', True)
                }

        # Handle bias quantizer
        bq = getattr(layer, 'bias_quantizer', None)
        if bq:
            if isinstance(bq, str):
                match = re.search(r'quantized_bits\((\d+),\s*(\d+)', bq)
                if match:
                    bits, integer = int(match.group(1)), int(match.group(2))
                    quant_config[layer_name]['biases'] = {
                        'bitwidth': bits,
                        'integer_bits': integer,
                        'signed': True
                    }
            elif hasattr(bq, 'bits') and hasattr(bq, 'integer'):
                quant_config[layer_name]['biases'] = {
                    'bitwidth': bq.bits,
                    'integer_bits': bq.integer,
                    'signed': getattr(bq, 'signed', True)
                }

        # Handle activation quantizer (QActivation layer or standalone activation layer)
        if layer.__class__.__name__ == "QActivation":
            qact = getattr(layer, 'activation', None)
            if isinstance(qact, str):
                match = re.search(r'quantized_relu\((\d+),\s*(\d+)', qact)
                if match:
                    bits = int(match.group(1))
                    integer = int(match.group(2))
                    quant_config[layer_name]['activations'] = {
                        'bitwidth': bits,
                        'integer_bits': integer,
                        'signed': False
                    }
            elif hasattr(qact, '__name__'):
                match = re.search(r'quantized_relu\((\d+),\s*(\d+)', qact.__name__)
                if match:
                    bits = int(match.group(1))
                    integer = int(match.group(2))
                    quant_config[layer_name]['activations'] = {
                        'bitwidth': bits,
                        'integer_bits': integer,
                        'signed': False
                    }

        # PRINT summary
        if verbose:
            print(f"\nLayer: {layer_name}")
            printed = False

            if 'weights' in quant_config[layer_name]:
                q = quant_config[layer_name]['weights']
                print(f"  - Weights:  {q['bitwidth']}-bit, int={q['integer_bits']}, signed={q['signed']}")
                printed = True
            if 'biases' in quant_config[layer_name]:
                q = quant_config[layer_name]['biases']
                print(f"  - Biases:   {q['bitwidth']}-bit, int={q['integer_bits']}, signed={q['signed']}")
                printed = True
            if 'activations' in quant_config[layer_name]:
                q = quant_config[layer_name]['activations']
                print(f"  - Activation: quantized_relu({q['bitwidth']},{q['integer_bits']}) -> Q{q['integer_bits']}.{q['bitwidth'] - q['integer_bits']}")
                printed = True

            if not printed:
                print("  No quantization found for this layer.")

    return quant_config


def apply_quant_to_hls_config(hls_config, quant_config, default_precision='ap_fixed<16,6>', verbose=True):
    for layer_name in hls_config['LayerName']:
        q = quant_config.get(layer_name, {})
        hls_config['LayerName'][layer_name] = {'Precision': {}}

        if 'weights' in q:
            bw, iw = q['weights']['bitwidth'], q['weights']['integer_bits']
            signed = '' if q['weights'].get('signed', True) else 'u'
            prec = f"ap_{signed}fixed<{bw},{iw}>"
            hls_config['LayerName'][layer_name]['Precision']['weight'] = prec
            if verbose:
                print(f"[{layer_name}] Set weight precision to {prec}")
        else:
            hls_config['LayerName'][layer_name]['Precision']['weight'] = default_precision
            if verbose:
                print(f"[{layer_name}] No quantized weights - using default: {default_precision}")

        if 'biases' in q:
            bw, iw = q['biases']['bitwidth'], q['biases']['integer_bits']
            signed = '' if q['biases'].get('signed', True) else 'u'
            prec = f"ap_{signed}fixed<{bw},{iw}>"
            hls_config['LayerName'][layer_name]['Precision']['bias'] = prec
            if verbose:
                print(f"[{layer_name}] Set bias precision to {prec}")

        if 'activations' in q:
            bw, iw = q['activations']['bitwidth'], q['activations']['integer_bits']
            signed = '' if q['activations'].get('signed', True) else 'u'
            prec = f"ap_{signed}fixed<{bw},{iw}>"
            hls_config['LayerName'][layer_name]['Precision']['result'] = prec
            if verbose:
                print(f"[{layer_name}] Set activation/output precision to {prec}")
        else:
            hls_config['LayerName'][layer_name]['Precision']['result'] = default_precision
            if verbose:
                print(f"[{layer_name}] No quantized activation - using default: {default_precision}")
