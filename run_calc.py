from CRL import CRL

if __name__ == '__main__':
    l = [2, 4, 6, 7, 8]
    e = 21500
    p0 = 6.52

    crl1 = CRL(cart_ids=l, energy=e, p0=p0, use_numpy=True)
    p1 = crl1.calc_real_lens()
    p1_ideal = crl1.calc_ideal_lens()

    crl2 = CRL(cart_ids=l, energy=e, p0=p0, use_numpy=False)
    p2 = crl2.calc_real_lens()
    p2_ideal = crl2.calc_ideal_lens()

    d = crl1.calc_delta_focus(p1)
    d_ideal = crl1.calc_delta_focus(p1_ideal)

    print('P0: {}, P1: {}, P1 ideal: {}, d: {}, d ideal: {}'.format(crl1.p0, p1, p1_ideal, d, d_ideal))
