def recurse(remaining, backup):
    print(remaining, backup)  
    if len(remaining) == 1:
       if 23.99999 <= remaining[0] and remaining[0] <= 24.000001:
           return backup
       else: return False

    for i in range(len(remaining)):
        for j in range(len(remaining)):
            if (i == j): continue
            leftover = []
            for k in range(len(remaining)):
                if k == i or k == j: continue
                else: leftover.append(remaining[k])

            doNOTTRY = False
            leftover1 = list(tuple(leftover))
            leftover2 = list(tuple(leftover))
            leftover3 = list(tuple(leftover))
            leftover4 = list(tuple(leftover))
            leftover1.append(remaining[i] + remaining[j])
            leftover2.append(remaining[i] * remaining[j])
            try: leftover3.append(remaining[i] / remaining[j])
            except: doNOTTRY = True
            leftover4.append(remaining[i] - remaining[j])
          
            backup1 = list(tuple(backup))
            backup2 = list(tuple(backup))
            backup3 = list(tuple(backup))
            backup4 = list(tuple(backup))
            backup1.append("add %s %s" % (remaining[i], remaining[j]))
            backup2.append("multiply %s %s" % (remaining[i], remaining[j]))
            backup3.append("divide %s %s" % (remaining[i], remaining[j]))
            backup4.append("subtract %s %s" % (remaining[i], remaining[j]))
          
            first = recurse(leftover1, backup1)
            second = recurse(leftover2, backup2)
            if (not doNOTTRY):
              third = recurse(leftover3, backup3)
              if (third): return third
            fourth = recurse(leftover4, backup4)
            
            if (first): return first
            if (second): return second
            if (fourth): return fourth

    return False

print(recurse([8.0,3.0,3.0,8.0], tuple()))
